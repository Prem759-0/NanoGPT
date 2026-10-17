"""
NanoGPT Lab — Terminal CLI Assistant
Run and chat with your local AI directly from PowerShell / Terminal!
Includes memory context, code execution sandbox, and image generation.
"""
import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

from webapp.model_service import ModelService, ModelNotTrainedException
from webapp.database import db
from webapp.image_service import ImageService
from webapp.code_service import CodeService

def main():
    print("=" * 65)
    print("   [NanoGPT Lab] -- Super AI Interactive Terminal Assistant       ")
    print("=" * 65)

    print("Commands:")
    print("  Type your prompt to chat with your local AI")
    print("  /model <gpt2|scratch> : Switch between GPT-2 124M and Scratch model")
    print("  /models               : List available AI engines")
    print("  /imagine <prompt>     : Generate procedural neural artwork")
    print("  /run <code>           : Run Python code in sandbox")
    print("  /history              : View recent conversation memory")
    print("  /exit or /quit        : Exit application")
    print("=" * 65)

    # Initialize services
    model_service = ModelService(base_dir=PROJECT_ROOT)
    output_dir = os.path.join(PROJECT_ROOT, "webapp", "static", "generated")
    image_service = ImageService(output_dir=output_dir)
    code_service = CodeService()

    status = model_service.get_status()
    print(f"\n[System Status]: Model is {status} on {model_service.device.upper()}")
    
    session_id = db.create_session(title="Terminal Chat Session")
    print(f"[Session ID]: {session_id} (Stored securely in local SQLite)\n")

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                print("\nGoodbye! Have a great day.")
                break

            # 0. Model Switcher Command
            if user_input.startswith("/model ") or user_input == "/models":
                if user_input == "/models":
                    models = model_service.get_available_models()
                    print("\n--- Available Local AI Engines ---")
                    for m in models:
                        active_flag = " [ACTIVE]" if m.get("active") else ""
                        print(f"• {m['id']}: {m['name']} ({m['parameters']} params){active_flag}")
                        print(f"  {m['description']}")
                    print("Switch with: /model llama-3.2  OR  /model gpt2  OR  /model scratch\n")
                    continue
                target_m = user_input.split(" ", 1)[1].strip().lower()
                if target_m not in ["llama-3.2", "llama", "gguf", "gpt2", "scratch"]:
                    print("⚠️ Invalid model. Choose '/model llama-3.2', '/model gpt2', or '/model scratch'")
                    continue
                if target_m in ["llama", "gguf"]:
                    target_m = "llama-3.2"
                print(f"🔄 Switching model to '{target_m}'...")
                success = model_service.switch_model(target_m)
                if success:
                    print(f"✅ Active model is now: {model_service.get_info()['model_type']}")
                else:
                    print(f"❌ Failed to switch: {model_service.error_message}")
                continue

            # 1. Image Generation Command
            if user_input.startswith("/imagine ") or user_input.startswith("/image "):
                prompt = user_input.split(" ", 1)[1].strip()
                print("🎨 Generating local artwork...")
                res = image_service.generate_image(prompt)
                print(f"✅ Image saved: {res['filename']}")
                print(f"📁 Path: {os.path.join(output_dir, res['filename'])}")
                continue

            # 2. Code Runner Command
            if user_input.startswith("/run "):
                code = user_input.split(" ", 1)[1].strip()
                print("⚡ Executing in local Python sandbox...")
                res = code_service.execute_code(code)
                print(f"[Exit code: {res['exit_code']} | Time: {res['execution_time_ms']}ms]")
                if res['output']:
                    print(res['output'])
                if res['error']:
                    print(f"Error:\n{res['error']}")
                continue

            # 3. View Memory History Command
            if user_input == "/history":
                messages = db.get_session_messages(session_id, limit=10)
                print(f"\n--- Conversation History ({len(messages)} turns) ---")
                for m in messages:
                    print(f"{m['role'].upper()}: {m['content']}")
                print("------------------------------------------")
                continue

            # 4. Neural Generation with Context Memory
            history = db.get_session_messages(session_id, limit=6)
            db.add_message(session_id, "user", user_input)

            try:
                print("AI is thinking...", end="\r", flush=True)
                result = model_service.generate(prompt=user_input, max_new_tokens=150, temperature=0.8, history=history)
                reply = result["reply"]
                db.add_message(session_id, "assistant", reply)
                print(" " * 30, end="\r")  # Clear thinking text
                print(f"AI ({result['model_name']}) >\n{reply}")
            except ModelNotTrainedException as e:
                print(f"\n[Error]: {e}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Data saved in SQLite.")
            break

if __name__ == "__main__":
    main()
