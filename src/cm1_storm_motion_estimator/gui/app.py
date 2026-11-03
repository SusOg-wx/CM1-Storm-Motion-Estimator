from __future__ import annotations

import sys

from PySide6 import QtWidgets

from .main_window import CM1MainWindow


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CM1MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
