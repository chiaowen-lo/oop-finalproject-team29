# oop-finalproject-team29 
> An implementation of a custom Reinforcement Learning environment using **Gymnasium**, **Pygame**, and **Q-Learning**, demonstrating advanced Object-Oriented Programming (OOP) principles.

## Project Overview (專案概述)
本專案是國立中山大學 114 學年度物件導向程式設計 (OOP) 的期末專案。我們旨在透過構建一個完整的 AI 遊戲系統，展示軟體工程中的模組化設計、封裝與繼承應用，並結合強化學習演算法解決複雜控制問題。

專案共分為三個階段，核心重點在於 Part 3 的自定義環境開發與混合式 AI 實作：

### Part 1: Environment Setup
* **目標**：建立並驗證 Gymnasium 與 Python 開發環境。
* **內容**：確保依賴套件 (Dependencies) 安裝正確，並能成功運行基礎 Gym 環境。

### Part 2: Frozen Lake Optimization (SARSA & Map Scavenging)
* **目標**：解決 `FrozenLake-v1` 8x8 (Slippery) 難題，達成 >0.70 的穩定成功率。
* **內容**：有別於傳統 Q-Learning，我們選擇實作 **SARSA (On-Policy)** 演算法以學習更安全的路徑，並開發**地圖篩選機制 (Map Scavenging)** 自動尋找可解且合理的隨機地圖。

### Part 3: Custom Snake AI (Core Project)
* **目標**：OOP 架構實作、自定義強化學習環境與混合式演算法 (Hybrid AI)。
* **內容**：我們從零打造了一個符合 Gymnasium 標準介面的貪吃蛇環境 (`SnakeEnv`)。
    * **OOP 設計**：將遊戲邏輯封裝為 `Snake` 與 `Food` 物件，並透過 `SnakeEnv` 繼承 `gym.Env` 進行抽象化管理。
    * **演算法整合**：結合 **Q-Learning** 進行自主學習，並在測試階段引入 **Flood Fill (BFS)** 演算法作為安全閥，防止 AI 走入死胡同。
    * **狀態優化**：設計了 11維相對視角 (Relative Observation) 狀態空間，解決了傳統座標輸入難以訓練的問題，並結合 Pygame 進行即時視覺化呈現。

## How to Run (如何執行)

本專案分為三個部分，請依序執行以下指令來驗證各階段成果。

### Part 1: Mountain Car
訓練並測試強化學習 Agent：

```bash
cd part1
# Train the agent
python mountain_car.py --train --episodes 5000

# Render and visualize performance
python mountain_car.py --render --episodes 10
```
### Part 2: Frozen Lake 
執行自動化地圖篩選與 SARSA 模型訓練。

```bash
cd Part2
# 程式將自動篩選地圖並進行訓練
python frozen_lake.py
```
### Part 3: Greedy Snake  
> **這部分展示了 OOP 架構與 Hybrid AI (RL + Search Algorithm) 的結合。**

#### 1. 執行主程式 (AI Demo)
執行以下指令即可完成「訓練 -> 測試 -> 繪圖 -> Demo」的完整自動化流程：

```bash
cd part3
python agent.py
```
結果：程式將自動開啟視窗展示訓練好的 AI 遊玩畫面，並在結束後生成比較圖表。

#### 2. 測試環境 (Human Mode)
若只想單獨測試環境邏輯（人類手動遊玩），可執行：

```bash
cd part3
python snake_env.py
```

### Part 3 Technical Details (技術細節)

#### 1. 系統架構 (OOP Architecture)
我們將系統拆分為三個核心物件：

* **`SnakeEnv` (Inherits `gym.Env`)**: 控制器，負責協調物件互動並計算 **11維相對視角狀態**。
* **`Snake` & `Food`**: 封裝移動、生長與重生邏輯，獨立於環境之外。

#### 2. 演算法機制 (Methodology)
* **Training (Q-Learning)**: 利用 11 維布林值狀態 (危險偵測、移動方向、食物方位) 進行高效率訓練。
* **Safety (Flood Fill Algorithm)**: 在測試階段，AI 決定動作前會先使用 **BFS** 計算連通空間。若空間 < 蛇身長度，AI 將強制避開該路徑。


## Dependencies (安裝與依賴)

本專案基於 **Python 3.8+** 開發。在執行 Part 3 之前，請確保已安裝下列 Python 套件：

```bash
pip install gymnasium numpy pygame matplotlib
```
## Contribution List (成員貢獻)

| 成員&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | 學號 | 負責項目 |
| :--- | :--- | :--- |
| <small>**羅巧雯**</small> | B123245020 | **Part 2**: 實作 SARSA 演算法與地圖篩選機制 (Map Scavenging)。<br>**Part 3**: 負責 Q-Learning 訓練邏輯、Flood Fill 安全機制實作 (`agent.py`) 與數據分析。 |
| <small>**林芷榆**</small> | B123245027 | **Part 3**: 負責 `SnakeEnv` 環境架構 (Gym Interface)、OOP 設計 (封裝/繼承)、11維相對視角狀態設計、以及技術文件 (UML + README)。 |
| <small>**楊子嫻**</small> | B123245029 | **Part 3**: 負責遊戲物件 (`Snake`, `Food`) 的邏輯封裝、 Pygame 視覺化呈現 (`render`)、Demo Slides (PDF) 以及 Reflection Report。 |
