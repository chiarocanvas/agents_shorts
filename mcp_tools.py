from typing import Any, Dict, List
from mcp.server.fastmcp import FastMCP
import yt_dlp
from pathlib import Path
import uuid
import whisper
import json
import os
import subprocess
import sys
import traceback
import tempfile
import shutil
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
def cut_video():
    

@mcp.tool()
def make_clips(path_to_transcribe:str) ->str:
    """Higlight  intresting  moments  from  video  and  clip  it """
    try:
        with  open(path_to_transcribe, "r", encoding="utf-8") as f:
            txt = json.load(f)
        payload_text = json.dumps(txt, ensure_ascii=False)
        result = call_llm_for_tool(tool_name="clipper", text=payload_text)
        data: Dict[str, Any] = result if isinstance(result, dict) else {"content": str(result)}
        project_dir = Path.cwd()
        clips_dir = project_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        clips_path = clips_dir / f"clips_{uuid.uuid4().hex}.json"
        with open(clips_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(clips_path)
    except Exception as e:
        # Сохраняем traceback для диагностики
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
