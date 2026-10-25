"""Default Hugging Face Spaces Entrypoint.
Imports and runs the Moon AI Cloud Engine from spaces_app.py.
"""
from spaces_app import demo, app

if __name__ == "__main__":
    demo.launch()
