# oop-finalproject-team29 
> An implementation of a custom Reinforcement Learning environment using **Gymnasium**, **Pygame**, and **Q-Learning**, demonstrating advanced Object-Oriented Programming (OOP) principles.

# Project Overview (專案概述)
本專案是國立中山大學 114 學年度物件導向程式設計 (OOP) 的期末專案。我們旨在透過構建一個完整的 AI 遊戲系統，展示軟體工程中的模組化設計、封裝與繼承應用。
專案共分為三個階段，核心重點在於Part 3 的自定義環境開發：

## Part 1: Environment Setup
* **目標**：建立並驗證 Gymnasium 與 Python 開發環境。
* **內容**：確保依賴套件 (Dependencies) 安裝正確，並能成功運行基礎 Gym 環境。

## Part 2: Frozen Lake Optimization
* **目標**：強化學習參數調校 (Hyperparameter Tuning)。
* **內容**：在不修改訓練回合數限制下，針對 `Frozen Lake-v1` 環境調整 Exploration Rate 與 Decay Rate，達成 >0.70 的穩定成功率。

## Part 3: Custom Snake AI (Core Project)
* **目標**：OOP 架構實作與自定義強化學習環境。
* **內容**：我們從零打造了一個符合 Gymnasium 標準介面的貪吃蛇環境 (`SnakeEnv`)。
    * **OOP 設計**：將遊戲邏輯封裝為 `Snake` 與 `Food` 物件，並透過 `SnakeEnv` 繼承 `gym.Env` 進行抽象化管理。
    * **演算法整合**：實作 Q-Learning 演算法 (`QLearningAgent`) 讓電腦自主學習。
    * **狀態優化**：設計了 11維相對視角 (Relative Observation) 狀態空間，解決了傳統座標輸入難以訓練的問題，並結合 Pygame 進行即時視覺化呈現。
