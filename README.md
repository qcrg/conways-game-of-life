# Implementation of John Conway's Game of Life

### Building
```bash
cd conways-game-of-life && cmake -DDISPLAY_TYPE=curses -DCMAKE_BUILD_TYPE=Release -Bbuild/ -S./ && cmake --build build/ -j8
```

### Dislpay Types
- curses
- sdl (not implemented)
- gtk (not implemented)
- qt (not implemented)

### Controls
- curses
```
h - left
j - down
k - up
l - right
p - play/pause
o - one step
- - decrease speed
= (+) - increase speed
```
- sdl (not implemented)
```
```
- gtk (not implemented)
```
```
- qt (not implemented)
```
```
