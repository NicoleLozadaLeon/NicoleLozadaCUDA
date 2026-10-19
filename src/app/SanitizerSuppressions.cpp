#ifdef DELIVERY_SIM_TSAN
extern "C" const char* __tsan_default_suppressions() {
    return "race:libQt5Core.so\n"
           "race:libQt5Gui.so\n"
           "race:libQt5Widgets.so\n"
           "race:libQt5DBus.so\n"
           "race:libQt5XcbQpa.so\n"
           "race:libdbus-1.so\n"
           "race:libglib-2.0.so\n"
           "race:libgobject-2.0.so\n"
           "race:libgio-2.0.so\n"
           "race:libgtk-3.so\n"
           "race:libgdk-3.so\n"
           "race:libX11.so\n"
           "race:libxcb.so\n"
           "deadlock:libQt5Core.so\n"
           "deadlock:libQt5DBus.so\n"
           "deadlock:libdbus-1.so\n"
           "deadlock:libglib-2.0.so\n"
           "deadlock:libgtk-3.so\n"
           "deadlock:libX11.so\n";
}
#endif

#ifdef DELIVERY_SIM_ASAN
extern "C" const char* __lsan_default_suppressions() {
    return "leak:libfontconfig.so\n"
           "leak:libQt5Core.so\n"
           "leak:libQt5Gui.so\n"
           "leak:libQt5Widgets.so\n"
           "leak:libQt5DBus.so\n"
           "leak:libQt5XcbQpa.so\n"
           "leak:libdbus-1.so\n"
           "leak:libglib-2.0.so\n"
           "leak:libgobject-2.0.so\n"
           "leak:libgtk-3.so\n"
           "leak:libgdk-3.so\n"
           "leak:libX11.so\n"
           "leak:libxcb.so\n";
}
#endif
