import sys
from PyQt5.QtWidgets import QApplication
from gui.main_menu import MainMenu
from gui.theme import apply_theme


def main():
    """Điểm khởi chạy chính (Entry point) của toàn bộ ứng dụng PyQt5."""
    # Khởi tạo instance application
    app = QApplication(sys.argv)
    
    # Thiết lập giao diện màu sắc / theme tổng thể
    apply_theme(app)

    window = MainMenu()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()