# Implementation of John Conway's Game of Life

### Building
```bash
cd conways-game-of-life
export WINDOW_TYPE=curses
cmake -DCMAKE_BUILD_TYPE=Release -Bbuild/ -S./ && cmake --build build/ -j8
```

### Dislpay Types
- curses
- sdl
- gtk (partial implementation)

### Controls
- Common
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
- curses
```
i - insert mode
space - change cell in insert mode
ESC - escape from insert mode
```
- GUI
```
middle button + mouse motion - screen movement
left button - change cell
mouse wheel - zoom
```
