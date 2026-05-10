from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

app = FastAPI()

# HTMLからアクセスできるようにする
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameState(BaseModel):
    board: list[list[Any]]
    playerHand: list[str]
    enemyHand: list[str]
    turn: str


@app.get("/")
def root():
    return {"message": "Shogi AI Backend Running"}


@app.post("/api/ai-move")
def ai_move(game_state: GameState):

    print("盤面を受信しました")
    print(game_state.turn)

    # 今はダミー応答
    return {
        "status": "ok",
        "message": "Python側で盤面を受信しました",
        "next_action": "ここにAIの指し手を返す"
    }