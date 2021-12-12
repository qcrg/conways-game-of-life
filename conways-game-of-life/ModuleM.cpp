#include "defines.h"

#include "ModuleM.h"

#pragma warning (disable: 26451 26495)

GameField::GameField(size_t w, size_t h)
    : maxSize{ w, h }
{
    allocMemForGameField();
}

GameField::GameField(Point p)
    : maxSize{ std::move(p) }
{
    allocMemForGameField();
}

GameField::~GameField()
{
    cuCHECK(cudaFreeHost(gameFieldData));
}

bool* GameField::data()
{
    return gameFieldData;
}

const bool* const GameField::data() const
{
    return gameFieldData;
}

bool& GameField::operator[](const Point& p)
{
    return gameFieldData[getIdx(p)];
}

const bool& GameField::operator[](const Point& p) const
{
    return gameFieldData[getIdx(p)];
}

const Point& GameField::getMaxSize() const
{
    return maxSize;
}

inline size_t GameField::getIdx(const Point& p) const
{
    return p.get<1>() * maxSize.get<1>() + p.get<0>();
}

void GameField::allocMemForGameField()
{
    cuCHECK(
        cudaMallocHost(&gameFieldData
            , sizeof(bool) * maxSize.get<0>() * maxSize.get<1>())
    );
}

bool Data::changeCell(Point cell)
{
    bool& cage = field[cell];
    if (cage)
    {
        aliveCells.erase(cell);
    }
    else
    {
        aliveCells.insert(std::move(cell));
    }
    cage = !cage;
    return !cage;
}