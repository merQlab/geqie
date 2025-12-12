from pathlib import Path

from geqie.cli import _import_module, ENCODINGS_PATH


def main():
    print("ENCODINGS_PATH:", ENCODINGS_PATH)

    # 1) Create a harmless module OUTSIDE encodings (simulating an attacker target)
    outside_path = ENCODINGS_PATH.parent / "evil_outside.py"
    outside_path.write_text(
        'print("EVIL MODULE EXECUTED")\n'
        'X = 42\n',
        encoding="utf-8",
    )
    print("Created outside module:", outside_path)

    # 2) Create a harmless module INSIDE encodings (legitimate plugin)
    inside_dir = ENCODINGS_PATH / "safe_plugin"
    inside_dir.mkdir(parents=True, exist_ok=True)
    inside_module = inside_dir / "init.py"
    inside_module.write_text(
        'print("SAFE PLUGIN LOADED")\n'
        'def init(image, **kwargs):\n'
        '    return image\n',
        encoding="utf-8",
    )
    print("Created inside module:", inside_module)

    # 3) First, verify that a normal, in-encodings import works
    print("\n[TEST] Importing safe_plugin/init.py (should SUCCEED)")
    mod_safe = _import_module("safe_init", "safe_plugin/init.py")
    print("Imported module:", mod_safe)

    # 4) Now, try to escape encodings with '../evil_outside.py'
    print("\n[TEST] Importing ../evil_outside.py (should FAIL)")
    try:
        _import_module("evil", "../evil_outside.py")
        print("ERROR: Path traversal WAS NOT prevented!")
    except ValueError as e:
        print("OK: Path traversal blocked with ValueError:")
        print("   ", e)
    except FileNotFoundError as e:
        # Also acceptable: we refused to resolve a valid file outside encodings
        print("OK: Path traversal blocked with FileNotFoundError:")
        print("   ", e)


if __name__ == "__main__":
    main()
