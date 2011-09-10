CFLAGS = -Wall -DXP_UNIX=1 -DMOZ_X11=1 -fPIC -g
INCLUDE = -I/home/home/xulrunner-sdk/include -I/usr/include/opencv -I/usr/src/linux-headers-2.6.35-22-generic/include/config/pci
CFLAGS += `pkg-config --cflags --libs gnome-keyring-1 glib-2.0`

npsimpleplugin : plugin.o
	cc $(CFLAGS) -shared -o npsimpleplugin.so \
np_entry.o npn_gate.o npp_gate.o plugin.o \
-lcv -lhighgui -lcxcore -lcvaux -lstdc++

plugin.o : plugin.cpp plugin.h
	cc -c $(CFLAGS) $(INCLUDE) $(LDFLAGS) \
np_entry.cpp npn_gate.cpp npp_gate.cpp plugin.cpp

