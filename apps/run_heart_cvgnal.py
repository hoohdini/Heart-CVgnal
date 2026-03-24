"""Entry point for Heart CV-gnal."""
from heart_cvgnal.app.runner import HeartCVgnalApp


def main() -> None:
    app = HeartCVgnalApp(camera_index=0)
    app.run()


if __name__ == "__main__":
    main()
