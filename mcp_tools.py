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
from moviepy import VideoFileClip
from  utils.call_llm import  call_llm_for_tool
from  utils.prompts import  _load_prompts




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
    

    
@mcp.tool()
def cut_video_segments(input_video_path: str, segments_json: str, output_dir: str = "segments") -> list:
    
    """
    MCP Tool: Нарезает видео на сегменты по временным меткам из JSON
    
    Args:
        input_video_path (str): Путь к исходному видеофайлу
        segments_json (str): JSON строка с сегментами ИЛИ путь к JSON файлу
        output_dir (str): Папка для сохранения сегментов (по умолчанию "segments")
    
    Returns:
        list: Список путей к созданным файлам сегментов
    
    """
    try:

        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Видеофайл не найден: {input_video_path}")
        
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
            raise ValueError(f"Неверный формат JSON. Ожидается: {{'segments': [{{'start': float, 'end': float}}, ...]}}. Ошибка: {str(e)}")
        
        os.makedirs(output_dir, exist_ok = True)




        os.environ['TQDM_DISABLE'] = '1'
        os.environ['TQDM_DISABLE_TTY'] = '1'

        video = VideoFileClip(input_video_path, audio=True)
        output_files = []
        
        try:
            base_name = os.path.splitext(os.path.basename(input_video_path))[0]
            
            for i, segment in enumerate(segments):
                try:
                    start_time = float(segment["start"]) 
                    end_time = float(segment["end"])      
                except Exception:
                    continue

                duration = float(getattr(video, "duration", 0.0) or 0.0)
                if duration and duration > 0:
                    start_time = max(0.0, min(start_time, duration))
                    end_time = max(0.0, min(end_time, duration))

                if end_time - start_time <= 0.05:
                    continue

                segment_clip = video.subclipped(start_time, end_time)
                

                output_filename = f"{base_name}_segment_{i+1:02d}_{start_time:.1f}-{end_time:.1f}s.mp4"
                output_path = os.path.join(output_dir, output_filename)
                

                has_audio = getattr(segment_clip, "audio", None) is not None


                ff_params = ["-map", "0:v:0"]
                if has_audio:
                    ff_params += ["-map", "0:a?"]

                    ff_params += ["-shortest"]


                segment_clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio=has_audio,
                    audio_codec=("aac" if has_audio else None),
                    audio_fps=(44100 if has_audio else None),
                    bitrate=None,
                    audio_bitrate=("192k" if has_audio else None),
                    fps=getattr(video, "fps", None) or 24,
                    ffmpeg_params=ff_params,
                    logger=None
                )
                
                output_files.append(output_path)
                segment_clip.close()
            
            return output_files
            
        finally:
            video.close(
            )
    except Exception as e:
        error_info = {
            "error": str(e),
            "error_type": type(e).__name__,
            "file_path": output_path,
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
def make_clips(path_to_transcribe:str) ->str:
    """Higlight  intresting  moments  from  video  and  clip  it """
    try:
        with  open(path_to_transcribe, "r", encoding="utf-8") as f:
            txt = json.load(f)
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
    """Download a video using yt_dlp and return an absolute file path as a string.

    Returns: "C:/abs/path/to/file.mp4"
    """
    try:
        project_dir = Path.cwd()
        downloads_dir = project_dir / "downloads"
        downloads_dir.mkdir(exist_ok=True)
        
        base_name = f"video_{uuid.uuid4().hex}"
        outtmpl = str(downloads_dir / f"{base_name}.%(ext)s")

        ydl_opts = {
            'outtmpl': outtmpl,
            'format': 'best[ext=mp4]/best', 
            'quiet': True,
            'no_warnings': True,
            'noprogress': True,
            'logger': _NullLogger(),
            'consoletitle': False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            try:
                file_path = ydl.prepare_filename(info)
            except Exception:
                file_path = str(downloads_dir / 'video.mp4')
        p = Path(file_path)
        if not p.exists():
            for ext in ("mp4", "mkv", "webm", "m4a", "mp3", "wav", "mov"):
                candidate = p.with_suffix(f".{ext}")
                if candidate.exists():
                    p = candidate
                    break
        return str(p.resolve().as_posix())
    except Exception as e:
        error_info = {
            "error": str(e),
            "error_type": type(e).__name__,
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
