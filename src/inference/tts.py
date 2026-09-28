"""Texto a voz offline en español de México, sin bloquear al que llama.

`speak()` regresa de inmediato: un hilo de fondo habla. Si llegan textos mientras
habla, solo se conserva el más reciente para que la voz no se atrase respecto a
las señas.

- macOS: comando `say` con settings.TTS_VOICE (Paulina, es_MX).
- Windows/Linux: pyttsx3 con la primera voz es-MX / es-419 / es que encuentre.
"""

import subprocess
import sys
import threading

from config import settings

_cond = threading.Condition()
_pending = None  # último texto sin decir
_busy = False
_worker = None


def label_to_text(label: str) -> str:
    return settings.SPOKEN_TEXT.get(label.lower(), label.replace("_", " "))


def say_label(label: str) -> None:
    """Dice una etiqueta del modelo (omite las de NON_SIGN_LABELS, p. ej. reposo)."""
    if label.lower() in settings.NON_SIGN_LABELS:
        return
    speak(label_to_text(label))


def speak(text: str) -> None:
    global _pending, _worker
    with _cond:
        _pending = text  # reemplaza lo que estuviera en espera
        if _worker is None:
            _worker = threading.Thread(target=_run, daemon=True)
            _worker.start()
        _cond.notify_all()


def wait() -> None:
    """Bloquea hasta que no quede nada por decir (útil al salir de un script)."""
    with _cond:
        _cond.wait_for(lambda: _pending is None and not _busy)


def _run() -> None:
    global _pending, _busy
    say = _say_macos if sys.platform == "darwin" else _make_pyttsx3_say()
    while True:
        with _cond:
            _cond.wait_for(lambda: _pending is not None)
            text, _pending, _busy = _pending, None, True
        try:
            say(text)
        except Exception as e:  # la voz nunca debe tumbar el reconocimiento
            print(f"[tts] error al hablar: {e}")
        with _cond:
            _busy = False
            _cond.notify_all()


def _say_macos(text: str) -> None:
    subprocess.run(["say", "-v", settings.TTS_VOICE, "-r", str(settings.TTS_RATE), text], check=True)


def _make_pyttsx3_say():
    # El motor se crea dentro del hilo que lo usa (pyttsx3 no es seguro entre hilos).
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", settings.TTS_RATE)
    voice = _pick_spanish_voice(engine.getProperty("voices"))
    if voice:
        engine.setProperty("voice", voice.id)
    else:
        print("[tts] AVISO: no se encontró voz en español; se usará la voz por defecto.")

    def say(text: str) -> None:
        engine.say(text)
        engine.runAndWait()

    return say


def _pick_spanish_voice(voices):
    def tags(v):
        langs = [x.decode(errors="ignore") if isinstance(x, bytes) else str(x) for x in (v.languages or [])]
        return " ".join([v.id, v.name or ""] + langs).lower().replace("_", "-")

    wanted = settings.TTS_LANG.lower().replace("_", "-")
    for preference in (wanted, "es-419", "es-"):
        for v in voices:
            if preference in tags(v) or (preference == "es-" and "spanish" in tags(v)):
                return v
    return None
