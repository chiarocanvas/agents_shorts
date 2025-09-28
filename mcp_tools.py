from typing import Any, Dict, List
from mcp.server.fastmcp import FastMCP
import yt_dlp
from pathlib import Path
import uuid
import whisper
import json
import os
import sys
import traceback
import tempfile
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from utils.call_llm import call_llm_for_tool
from utils.prompts import _load_prompts
import math

mcp = FastMCP("video_agent")


class _NullLogger:
    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


def create_subtitles_for_segment(segment_text: str, video_duration: float, 
                               video_size: tuple = (1080, 1920)) -> TextClip:
    """
    Создает субтитры для сегмента видео (оптимизировано для YouTube Shorts)
    
    Args:
        segment_text: Текст субтитров
        video_duration: Длительность видео в секундах
        video_size: Размер видео (ширина, высота)
    
    Returns:
        TextClip: Клип с субтитрами
    """
    # Настройки для YouTube Shorts (вертикальное видео)
    font_size = min(video_size) // 25  # Адаптивный размер шрифта
    
    # Разбиваем текст на слова для лучшего отображения
    words = segment_text.split()
    
    # Ограничиваем количество слов в строке (для читаемости)
    max_words_per_line = 4
    lines = []
    for i in range(0, len(words), max_words_per_line):
        line = ' '.join(words[i:i + max_words_per_line])
        lines.append(line)
    
    subtitle_text = '\n'.join(lines)
    
    # Создаем текстовый клип с обводкой
    txt_clip = TextClip(
        subtitle_text,
        fontsize=font_size,
        color='white',
        font='Arial-Bold',
        stroke_color='black',
        stroke_width=2,
        method='caption',
        size=(video_size[0] * 0.9, None),  # 90% ширины экрана
        align='center'
    ).set_duration(video_duration)
    
    # Позиционируем субтитры внизу экрана (но не слишком низко)
    txt_clip = txt_clip.set_position(('center', video_size[1] * 0.75))
    
    return txt_clip


def split_transcript_for_long_video(transcript_data: dict, max_segments: int = 50) -> List[dict]:
    """
    Разбивает длинные видео на части для обработки LLM
    
    Args:
        transcript_data: Полная транскрипция
        max_segments: Максимальное количество сегментов в одной части
    
    Returns:
        List[dict]: Список частей транскрипции
    """
    segments = transcript_data.get("segments", [])
    
    if len(segments) <= max_segments:
        return [transcript_data]
    
    parts = []
    for i in range(0, len(segments), max_segments):
        part_segments = segments[i:i + max_segments]
        
        # Создаем текст для этой части
        part_text = " ".join([seg.get("text", "") for seg in part_segments])
        
        part_data = {
            "text": part_text,
            "segments": part_segments,
            "video_path": transcript_data.get("video_path"),
            "part_number": (i // max_segments) + 1,
            "total_parts": math.ceil(len(segments) / max_segments)
        }
        parts.append(part_data)
    
    return parts


@mcp.tool()
def cut_video_segments(input_video_path: str, segments_json: str, 
                      output_dir: str = "segments", add_subtitles: bool = True) -> list:
    """
    MCP Tool: Нарезает видео на сегменты по временным меткам из JSON с возможностью добавления субтитров
    """
    try:
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Видеофайл не найден: {input_video_path}")
        
        # Парсинг JSON (без изменений)
        try:
            if not segments_json or str(segments_json).strip() == "":
                raise ValueError("JSON данные пустые")
            
            if isinstance(segments_json, dict):
                segments_data = segments_json
            else:
                segments_json_str = str(segments_json)
                
                if os.path.exists(segments_json_str) and segments_json_str.endswith('.json'):
                    with open(segments_json_str, 'r', encoding='utf-8') as f:
                        segments_data = json.load(f)
                else:
                    segments_data = json.loads(segments_json_str)
            
            if "segments" not in segments_data:
                raise ValueError("JSON не содержит ключ 'segments'")
            
            segments = segments_data["segments"]
            
            if not isinstance(segments, list):
                raise ValueError("segments должен быть списком")
            
            if len(segments) == 0:
                raise ValueError("Список segments пустой")
                    
        except (json.JSONDecodeError, KeyError, ValueError, FileNotFoundError) as e:
            raise ValueError(f"Неверный формат JSON. Ошибка: {str(e)}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # КРИТИЧНО: Перенаправляем все потоки вывода
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Создаем временные файлы для вывода
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_stdout, \
             tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_stderr:
            
            sys.stdout = temp_stdout
            sys.stderr = temp_stderr
            
            try:
                os.environ['TQDM_DISABLE'] = '1'
                os.environ['TQDM_DISABLE_TTY'] = '1'

                video = VideoFileClip(input_video_path, audio=True)
                output_files = []
                base_name = os.path.splitext(os.path.basename(input_video_path))[0]
                video_size = (video.w, video.h)
                
                for i, segment in enumerate(segments):
                    try:
                        start_time = float(segment["start"]) 
                        end_time = float(segment["end"])
                        segment_text = segment.get("text", "").strip()
                    except Exception:
                        continue

                    duration = float(getattr(video, "duration", 0.0) or 0.0)
                    if duration and duration > 0:
                        start_time = max(0.0, min(start_time, duration))
                        end_time = max(0.0, min(end_time, duration))

                    if end_time - start_time <= 0.05:
                        continue

                    try:
                        segment_clip = video.subclipped(start_time, end_time)
                        clip_duration = end_time - start_time
                        
                        # Безопасно обрабатываем аудио
                        has_audio = False
                        try:
                            if hasattr(segment_clip, 'audio') and segment_clip.audio is not None:
                                segment_clip.audio.get_frame(0)
                                has_audio = True
                        except:
                            segment_clip = segment_clip.without_audio()
                            has_audio = False
                        
                        # Добавляем субтитры
                        if add_subtitles and segment_text:
                            try:
                                subtitle_clip = create_subtitles_for_segment(
                                    segment_text, clip_duration, video_size
                                )
                                segment_clip = CompositeVideoClip([segment_clip, subtitle_clip])
                            except Exception:
                                pass  # Продолжаем без субтитров
                        
                        output_filename = f"{base_name}_segment_{i+1:02d}_{start_time:.1f}-{end_time:.1f}s.mp4"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # Простые параметры для экспорта
                        segment_clip.write_videofile(
                            output_path,
                            codec="libx264",
                            audio=has_audio,
                            audio_codec="aac" if has_audio else None,
                            fps=getattr(video, "fps", None) or 24,
                            logger=None,
                            verbose=False
                        )
                        
                        output_files.append(output_path)
                        segment_clip.close()
                        
                    except Exception:
                        continue  # Пропускаем проблемный сегмент
                
                video.close()
                return output_files
                
            finally:
                # КРИТИЧНО: Восстанавливаем потоки вывода
                sys.stdout = original_stdout  
                sys.stderr = original_stderr
                
                # Удаляем временные файлы
                try:
                    os.unlink(temp_stdout.name)
                    os.unlink(temp_stderr.name)
                except:
                    pass
            
    except Exception as e:
        error_info = {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        
        project_dir = Path.cwd()
        transcripts_dir = project_dir / "clips" 
        transcripts_dir.mkdir(exist_ok=True)
        
        error_path = transcripts_dir / f"clips_error_{uuid.uuid4().hex}.json"
        
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2)
        
        return str(error_path)


@mcp.tool()
def make_clips_chunked(path_to_transcribe: str, chunk_size: int = 50) -> str:
    """
    Обрабатывает длинные видео частями для избежания переполнения контекста LLM
    
    Args:
        path_to_transcribe: Путь к файлу транскрипции
        chunk_size: Размер чанка (количество сегментов)
    
    Returns:
        str: Путь к объединенному файлу с клипами
    """
    try:
        with open(path_to_transcribe, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        
        # Разбиваем на части
        parts = split_transcript_for_long_video(transcript_data, chunk_size)
        
        all_clips = []
        
        for part in parts:
            # Обрабатываем каждую часть отдельно
            payload_text = json.dumps(part, ensure_ascii=False)
            result = call_llm_for_tool(tool_name="clipper", text=payload_text)
            
            if isinstance(result, dict):
                data = result
            else:
                try:
                    data = json.loads(str(result))
                except json.JSONDecodeError:
                    data = {"content": str(result)}
            
            # Добавляем клипы из этой части
            if data and "segments" in data:
                part_clips = data["segments"]
                # Добавляем информацию о части
                for clip in part_clips:
                    clip["source_part"] = part["part_number"]
                all_clips.extend(part_clips)
        
        # Объединяем все клипы
        final_data = {
            "segments": all_clips,
            "video_path": transcript_data.get("video_path"),
            "total_parts_processed": len(parts),
            "original_segments_count": len(transcript_data.get("segments", []))
        }
        
        project_dir = Path.cwd()
        clips_dir = project_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        clips_path = clips_dir / f"clips_chunked_{uuid.uuid4().hex}.json"
        
        with open(clips_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        return str(clips_path)
        
    except Exception as e:
        error_info = {
            "error": str(e),
            "error_type": type(e).__name__,
            "file_path": path_to_transcribe,
            "traceback": traceback.format_exc()
        }
        
        project_dir = Path.cwd()
        transcripts_dir = project_dir / "clips" 
        transcripts_dir.mkdir(exist_ok=True)
        
        error_path = transcripts_dir / f"clips_error_{uuid.uuid4().hex}.json"
        
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2)
        
        print(f"Error saved to: {error_path}")
        return str(error_path)


@mcp.tool()
def make_clips(path_to_transcribe: str) -> str:
    """Highlight interesting moments from video and clip it (legacy function)"""
    try:
        with open(path_to_transcribe, "r", encoding="utf-8") as f:
            txt = json.load(f)
        
        # Проверяем размер транскрипции
        segments_count = len(txt.get("segments", []))
        
        # Если сегментов много, используем chunked версию
        if segments_count > 100:
            print(f"Длинное видео ({segments_count} сегментов), используем chunked обработку")
            return make_clips_chunked(path_to_transcribe, chunk_size=50)
        
        # Иначе обрабатываем как обычно
        payload_text = json.dumps(txt, ensure_ascii=False)
        result = call_llm_for_tool(tool_name="clipper", text=payload_text)
        
        if isinstance(result, dict):
            data = result
        else:
            try:
                data = json.loads(str(result))
            except json.JSONDecodeError:
                data = {"content": str(result)}
        
        if not data:
            raise ValueError("LLM вернул пустой результат")
        
        project_dir = Path.cwd()
        clips_dir = project_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        clips_path = clips_dir / f"clips_{uuid.uuid4().hex}.json"
        
        with open(clips_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return str(clips_path)
        
    except Exception as e:
        error_info = {
            "error": str(e),
            "error_type": type(e).__name__,
            "file_path": path_to_transcribe,
            "traceback": traceback.format_exc()
        }
        
        project_dir = Path.cwd()
        transcripts_dir = project_dir / "clips" 
        transcripts_dir.mkdir(exist_ok=True)
        
        error_path = transcripts_dir / f"clips_error_{uuid.uuid4().hex}.json"
        
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2)
        
        print(f"Error saved to: {error_path}")
        return str(error_path)


@mcp.tool()
def transcribe(path: str) -> str:
    """Transcribe an audio/video file with Whisper and return text plus segments."""
    try:
        model = whisper.load_model("small")
        os.environ['TQDM_DISABLE'] = '1'
        os.environ['TQDM_DISABLE_TTY'] = '1'
        
        original_stderr = sys.stderr
        stderr_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        sys.stderr = stderr_file
        
        try:
            result = model.transcribe(path, verbose=False)
        finally:
            sys.stderr = original_stderr
            stderr_file.close()
            os.unlink(stderr_file.name)

        text: str = result.get("text", "").strip()
        segments_src = result.get("segments") or []
        segments: List[Dict[str, Any]] = []
        
        for seg in segments_src:
            segments.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": (seg.get("text") or "").strip(),
            })

        project_dir = Path.cwd()
        transcripts_dir = project_dir / "transcripts"
        transcripts_dir.mkdir(exist_ok=True)
        
        out_path = transcripts_dir / f"transcript_{uuid.uuid4().hex}.json"
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "text": text, 
                "segments": segments,
                "video_path": path
            }, f, ensure_ascii=False, indent=2)

        return str(out_path)
                
    except Exception as e:
        error_info = {
            "error": str(e),
            "error_type": type(e).__name__,
            "file_path": path,
            "traceback": traceback.format_exc()
        }
        
        project_dir = Path.cwd()
        transcripts_dir = project_dir / "transcripts" 
        transcripts_dir.mkdir(exist_ok=True)
        
        error_path = transcripts_dir / f"transcript_error_{uuid.uuid4().hex}.json"
        
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2)
        
        print(f"Error saved to: {error_path}")
        return str(error_path)


@mcp.tool()
def get_video(url: str) -> str:
    """Download a video using yt_dlp and return an absolute file path as a string."""
    try:
        project_dir = Path.cwd()
        downloads_dir = project_dir / "downloads"
        downloads_dir.mkdir(exist_ok=True)
        
        base_name = f"video_{uuid.uuid4().hex}"
        outtmpl = str(downloads_dir / f"{base_name}.%(ext)s")

        # Простые варианты настроек (от самого простого)
        configs = [
            {'format': None},  # Без указания формата вообще
            {'format': 'worst'},  # Худшее качество
            {'format': 'bestaudio/best'},  # Только аудио в крайнем случае
        ]

        for config in configs:
            try:
                ydl_opts = {
                    'outtmpl': outtmpl,
                    'quiet': True,
                    'no_warnings': True,
                    'noprogress': True,
                    'logger': _NullLogger(),
                    'consoletitle': False,
                }
                
                # Добавляем формат только если он указан
                if config['format']:
                    ydl_opts['format'] = config['format']
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    file_path = ydl.prepare_filename(info)
                    
                p = Path(file_path)
                if not p.exists():
                    # Ищем любой скачанный файл
                    for ext in ("mp4", "mkv", "webm", "m4a", "mp3", "wav", "mov"):
                        candidate = p.with_suffix(f".{ext}")
                        if candidate.exists():
                            p = candidate
                            break
                
                if p.exists():
                    return str(p.resolve().as_posix())
                    
            except Exception:
                continue  # Пробуем следующий конфиг
        
        raise Exception(f"Не удалось скачать видео: {url}")
        
    except Exception as e:
        error_info = {
            "error": str(e),
            "error_type": type(e).__name__,
            "url": url,
            "traceback": traceback.format_exc()
        }
        
        project_dir = Path.cwd()
        errors_dir = project_dir / "errors" 
        errors_dir.mkdir(exist_ok=True)
        
        error_path = errors_dir / f"get_video_error_{uuid.uuid4().hex}.json"
        
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2)
        
        print(f"Error saved to: {error_path}")
        return str(error_path)


if __name__ == "__main__":
    mcp.run(transport='stdio')