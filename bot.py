import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP, WebSocket
import os
from dotenv import load_dotenv
import time
import logging
from logging.handlers import RotatingFileHandler
import warnings
from pandas_ta import rsi, atr, macd, adx
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import threading
import tensorflow as tf
import gc
import requests
from sklearn.preprocessing import MinMaxScaler

# Включение Eager Execution и отладочного режима для tf.data
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Отключение GPU и снижение уровня логов TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.config.set_visible_devices([], "GPU")

# Отключение предупреждений pandas_ta
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.core")
# Отключение предупреждений о downcasting
pd.set_option('future.no_silent_downcasting', True)

# Логирование с ротацией
logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Уменьшено с INFO до WARNING
handler = RotatingFileHandler("bot.log", maxBytes=100_000, backupCount=2)  # Уменьшен размер файла
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logging.critical("Скрипт запущен")

# Загрузка API ключей
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    logging.critical("API_KEY или API_SECRET не найдены в .env")
    raise ValueError("API_KEY или API_SECRET не найдены в .env")

# Подключение к Bybit API
session = HTTP(testnet=False, api_key=API_KEY, api_secret=API_SECRET)
logging.critical("Подключение к Bybit API успешно")

# Торговые параметры
SYMBOL = "DOGEUSDT"
LEVERAGE = 25  # Плечо 25x
FIXED_BALANCE = 1.0  # Фиксированный баланс для торговли (изменено с 10 на 1 USDT)
MIN_QUANTITY = 100  # Минимальный размер ордера для DOGEUSDT
TICK_SIZE = 0.00001
ATR_PERIOD = 20
ATR_MULTIPLIER_SL = 2.0  # Уменьшено для SL
ATR_MULTIPLIER_TP = 4.0  # Увеличено для TP
MIN_SL_DISTANCE = 0.01
MIN_TP_DISTANCE = 0.012
MIN_TRADE_INTERVAL = 900  # 15 минут
RETRAIN_INTERVAL = 3 * 60 * 60  # Дообучение каждые 3 часа
MAX_DF_ROWS = 100
MIN_ATR = 0.0001
MIN_ADX = 20  # Минимальный ADX для исключения флэта
# --- Risk toggles ---
USE_HARD_SL = False  # False = ставим только TP, SL не отправляем в API
SOFT_RISK_CUTOFF_ATR = 0.0  # 0 чтобы отключить; напр. 8.0 = аварийный выход по 8*ATR
MAX_HOLD_SECONDS = 0  # 0 чтобы отключить тайм-аут удержания позиции

time_steps = 20
features = [
    "open", "high", "low", "close", "volume", "turnover", "rsi", "atr",
    "macd", "macd_signal", "macd_hist", "adx", "ema_diff", "trend"
]
TAKER_FEE = 0.00075  # Комиссия тейкера на Bybit

# Инициализация DataFrame
df = pd.DataFrame(columns=["timestamp"] + features + ["ema_fast", "ema_slow", "volume_sma"])
for col in features + ["ema_fast", "ema_slow", "volume_sma"]:
    if col == "trend":
        df[col] = df[col].astype(int)
    else:
        df[col] = df[col].astype(float)
df["timestamp"] = pd.to_datetime(df["timestamp"]) 

# Глобальные переменные
position_type = None
entry_price = 0.0
take_profit_price = 0.0
stop_loss_price = 0.0
successful_trades = 0
total_trades = 0
last_trade_time = 0
last_message_time = time.time()
retrain_running = False
loss_streak = 0
pause_until = 0
data_buffer = []
last_retrain_time = 0

# Утилиты устойчивости для timestamp и индикаторов
def ensure_timestamp_column(df_in: pd.DataFrame) -> pd.DataFrame:
    try:
        if "timestamp" not in df_in.columns:
            # Пытаемся восстановить из индекса
            try:
                df_in = df_in.reset_index()
            except Exception:
                pass
            if "timestamp" not in df_in.columns and "index" in df_in.columns:
                df_in = df_in.rename(columns={"index": "timestamp"})
        return df_in
    except Exception as e:
        logging.critical(f"ensure_timestamp_column: {e}")
        return df_in


def update_indicators_safely(df_in: pd.DataFrame) -> pd.DataFrame:
    """Рассчитать индикаторы на копии и гарантировать возврат колонки timestamp."""
    df_work = None
    try:
        df_work = df_in.copy()
        df_work = ensure_timestamp_column(df_work)
        df_work = df_work.sort_values(by="timestamp")
        df_work.set_index("timestamp", inplace=True)

        df_work["rsi"] = rsi(df_work["close"], length=14)
        df_work["atr"] = atr(df_work["high"], df_work["low"], df_work["close"], length=ATR_PERIOD)
        macd_df_local = macd(df_work["close"], fast=12, slow=26, signal=9)
        if macd_df_local is None or not all(col in macd_df_local for col in ["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"]):
            raise ValueError("MACD вернул None или некорректный DataFrame")
        df_work["macd"] = macd_df_local["MACD_12_26_9"]
        df_work["macd_signal"] = macd_df_local["MACDs_12_26_9"]
        df_work["macd_hist"] = macd_df_local["MACDh_12_26_9"]
        adx_result_local = adx(df_work["high"], df_work["low"], df_work["close"], length=14)
        df_work["adx"] = adx_result_local["ADX_14"] if adx_result_local is not None and "ADX_14" in adx_result_local else 0.0
        df_work["ema_fast"] = df_work["close"].ewm(span=20, adjust=False).mean()
        df_work["ema_slow"] = df_work["close"].ewm(span=50, adjust=False).mean()
        df_work["ema_diff"] = df_work["ema_fast"] - df_work["ema_slow"]
        df_work["trend"] = np.where(df_work["ema_fast"] > df_work["ema_slow"], 1, -1)
        df_work["volume_sma"] = df_work["volume"].rolling(window=20).mean()

        df_work.reset_index(inplace=True)
        # Заполняем NaN после вычислений
        cols_fill = ["rsi", "atr", "macd", "macd_signal", "macd_hist", "adx", "ema_diff", "ema_fast", "ema_slow", "volume_sma"]
        df_work[cols_fill] = df_work[cols_fill].fillna(0)
        df_work["trend"] = df_work["trend"].fillna(0).astype(int)
        return df_work
    except Exception as e:
        logging.critical(f"Ошибка расчета индикаторов: {e}")
        # Гарантировать возврат с колонкой timestamp
        df_safe = df_in.copy()
        df_safe = ensure_timestamp_column(df_safe)
        return df_safe
    finally:
        # На всякий случай: не оставлять индекс timestamp
        try:
            if df_work is not None and isinstance(df_work.index, pd.DatetimeIndex):
                df_work.reset_index(inplace=True)
        except Exception:
            pass


# Создание scaler
try:
    scaler = joblib.load("scaler.pkl")
    if scaler.n_features_in_ != len(features):
        logging.warning(f"Scaler ожидает {scaler.n_features_in_} признаков, но текущий список имеет {len(features)} признаков. Создаем новый scaler.")
        scaler = MinMaxScaler()
        joblib.dump(scaler, "scaler.pkl")
except FileNotFoundError:
    logging.info("Scaler не найден, создаем новый")
    scaler = MinMaxScaler()
    joblib.dump(scaler, "scaler.pkl")

# Загрузка или создание модели с увеличенным Dropout
try:
    model = load_model("trading_model.h5")
    logging.critical("Модель загружена")
    expected_input_shape = (None, time_steps, len(features))
    if model.input_shape != expected_input_shape:
        logging.critical(f"Модель ожидает входную форму {model.input_shape}, но требуется {expected_input_shape}. Создаем новую модель.")
        raise ValueError("Несоответствие входной формы модели")
except Exception as e:
    logging.critical(f"Ошибка загрузки модели: {e}. Создаем новую модель.")
    model = Sequential([
        LSTM(128, input_shape=(time_steps, len(features)), return_sequences=True),
        Dropout(0.3),  # Увеличено с 0.2 до 0.3
        LSTM(64),
        Dropout(0.3),  # Увеличено с 0.2 до 0.3
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.save("trading_model.h5")
    logging.critical("Новая модель создана и сохранена")

# Функция для повторных попыток API-запросов
def api_request_with_retry(func, max_attempts=3, delay=5, **kwargs):
    for attempt in range(max_attempts):
        try:
            response = func(**kwargs)
            if response["retCode"] == 0:
                return response
            else:
                logging.critical(f"Ошибка API (попытка {attempt + 1}): {response.get('retMsg', 'Unknown error')}, retCode: {response['retCode']}")
                if attempt < max_attempts - 1:
                    time.sleep(delay)
        except requests.exceptions.RequestException as e:
            logging.critical(f"Ошибка сети (попытка {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(delay)
    return {"retCode": -1, "retMsg": "Максимальное количество попыток превышено"}

# Проверка API ключей
def check_api_keys():
    try:
        response = api_request_with_retry(
            session.get_wallet_balance,
            accountType="UNIFIED"
        )
        if response["retCode"] == 0:
            logging.critical("API ключи валидны, баланс получен")
            return True
        else:
            logging.critical(f"Ошибка проверки API ключей: {response.get('retMsg', 'Unknown error')}")
            return False
    except Exception as e:
        logging.critical(f"Ошибка проверки API ключей: {e}")
        return False

if not check_api_keys():
    logging.critical("Прекращение работы из-за невалидных API ключей")
    exit(1)

# Получение минимального шага цены
def get_tick_size(symbol):
    try:
        response = api_request_with_retry(
            session.get_instruments_info,
            category="linear",
            symbol=symbol
        )
        if response["retCode"] == 0:
            tick_size = float(response["result"]["list"][0]["priceFilter"]["tickSize"])
            logging.info(f"Минимальный шаг цены для {symbol}: {tick_size}")
            return tick_size
        else:
            logging.error(f"Ошибка получения tick_size: {response.get('retMsg', 'Unknown error')}")
            return TICK_SIZE
    except Exception as e:
        logging.error(f"Ошибка при получении tick_size: {e}")
        return TICK_SIZE

# Получение минимального количества
def get_min_quantity(symbol):
    try:
        response = api_request_with_retry(
            session.get_instruments_info,
            category="linear",
            symbol=symbol
        )
        if response["retCode"] == 0:
            min_qty = float(response["result"]["list"][0]["lotSizeFilter"]["minOrderQty"])
            logging.info(f"Минимальное количество для {symbol}: {min_qty}")
            return min_qty
        else:
            logging.error(f"Ошибка получения min quantity: {response.get('retMsg', 'Unknown error')}")
            return MIN_QUANTITY
    except Exception as e:
        logging.error(f"Ошибка при получении min_quantity: {e}")
        return MIN_QUANTITY

TICK_SIZE = get_tick_size(SYMBOL)
MIN_QUANTITY = get_min_quantity(SYMBOL)

# Установка плеча
def set_leverage(symbol, leverage):
    try:
        position_info = api_request_with_retry(
            session.get_positions,
            category="linear",
            symbol=symbol
        )
        if position_info["retCode"] == 0 and position_info["result"]["list"]:
            current_leverage = float(position_info["result"]["list"][0]["leverage"])
            logging.info(f"Текущее плечо для {symbol}: {current_leverage}x")
            if current_leverage == leverage:
                logging.info(f"Плечо {leverage}x уже установлено для {symbol}")
                return True

        response = api_request_with_retry(
            session.set_leverage,
            category="linear",
            symbol=symbol,
            buyLeverage=str(leverage),
            sellLeverage=str(leverage)
        )
        if response["retCode"] == 0 or response["retCode"] == 110043:
            logging.info(f"Плечо {leverage}x установлено для {symbol} (или уже установлено)")
            return True
        else:
            logging.critical(f"Ошибка установки плеча: {response.get('retMsg', 'Unknown error')}, retCode: {response['retCode']}")
            return False
    except Exception as e:
        logging.critical(f"Ошибка при установке плеча: {e}")
        return False

# Форматирование цены
def format_price(price, tick_size):
    try:
        return round(price / tick_size) * tick_size
    except Exception as e:
        logging.critical(f"Ошибка форматирования цены: {e}")
        return price

# Установка TP/SL (без трейлинг-стопа) — теперь может работать в режиме только TP
def set_trading_stop(position_type, entry_price, take_profit_price, stop_loss_price, quantity):
    try:
        formatted_take_profit = format_price(take_profit_price, TICK_SIZE)
        params = {
            "category": "linear",
            "symbol": SYMBOL,
            "takeProfit": str(formatted_take_profit),
            "positionIdx": 0,
            "isIsolated": True
        }
        if USE_HARD_SL and stop_loss_price and stop_loss_price > 0:
            formatted_stop_loss = format_price(stop_loss_price, TICK_SIZE)
            params["stopLoss"] = str(formatted_stop_loss)
        for attempt in range(3):
            response = api_request_with_retry(session.set_trading_stop, **params)
            if response["retCode"] == 0:
                if "stopLoss" in params:
                    logging.info(f"TP/SL установлены: TP={params['takeProfit']}, SL={params['stopLoss']}")
                else:
                    logging.info(f"TP установлен: TP={params['takeProfit']} (SL отключен)")
                return True
            logging.critical(f"Попытка {attempt + 1} установки TP/SL не удалась: {response.get('retMsg', 'Unknown error')}")
            time.sleep(5)
        logging.critical("Не удалось установить TP/SL после 3 попыток")
        return False
    except Exception as e:
        logging.critical(f"Ошибка при установке TP/SL: {e}")
        return False

# Проверка открытых ордеров
def check_open_orders(symbol):
    try:
        response = api_request_with_retry(
            session.get_open_orders,
            category="linear",
            symbol=symbol
        )
        if response["retCode"] == 0:
            open_orders = response["result"]["list"]
            if open_orders:
                logging.info(f"Найдено {len(open_orders)} открытых ордеров: {open_orders}")
            else:
                logging.info("Открытых ордеров нет")
            return open_orders
        else:
            logging.critical(f"Ошибка проверки открытых ордеров: {response.get('retMsg', 'Unknown error')}")
            return []
    except Exception as e:
        logging.critical(f"Ошибка при проверке открытых ордеров: {e}")
        return []

# Размещение ордера
def place_order(signal, symbol=SYMBOL, quantity=MIN_QUANTITY, current_price=0.0):
    try:
        if not set_leverage(symbol, LEVERAGE):
            logging.critical("Не удалось установить плечо")
            return None

        # Проверка текущей цены
        if current_price <= 0:
            logging.critical("Текущая цена <= 0, невозможно рассчитать размер позиции")
            return None

        # Проверка баланса
        balance_response = api_request_with_retry(session.get_wallet_balance, accountType="UNIFIED")
        if balance_response["retCode"] != 0:
            logging.critical(f"Ошибка получения баланса: {balance_response.get('retMsg', 'Unknown error')}, retCode: {balance_response['retCode']}")
            return None
        try:
            balance_data = balance_response["result"]["list"][0]
            logging.info(f"Полный ответ баланса: {balance_data}")
            total_equity = float(balance_data.get("totalEquity", 0))
            logging.info(f"Текущий баланс: TotalEquity={total_equity:.4f} USDT")
            if total_equity < FIXED_BALANCE:
                logging.critical(f"Недостаточный баланс: {total_equity:.4f} USDT, требуется минимум {FIXED_BALANCE} USDT")
                return None
        except (IndexError, KeyError, ValueError) as e:
            logging.critical(f"Ошибка обработки ответа баланса: {e}, Полный ответ: {balance_response}")
            return None

        # Проверка открытых ордеров
        open_orders = check_open_orders(symbol)
        if open_orders:
            logging.warning("Обнаружены открытые ордера, возможно, часть баланса заморожена")

        # Расчет quantity с учетом комиссии
        factor = (1 / LEVERAGE + TAKER_FEE * 2)
        quantity = int(FIXED_BALANCE / (current_price * factor))
        quantity = max(quantity, MIN_QUANTITY)
        position_value = quantity * current_price
        required_margin = position_value / LEVERAGE
        commission = position_value * TAKER_FEE * 2
        total_required = required_margin + commission

        # Финальная проверка
        if total_required > FIXED_BALANCE:
            logging.warning(f"Требуемая сумма {total_required:.4f} USDT превышает фиксированный баланс {FIXED_BALANCE} USDT, уменьшаем quantity")
            quantity = int(FIXED_BALANCE / (current_price * factor)) - 1
            quantity = max(quantity, MIN_QUANTITY)
            position_value = quantity * current_price
            required_margin = position_value / LEVERAGE
            commission = position_value * TAKER_FEE * 2
            total_required = required_margin + commission

        if total_required > total_equity:
            logging.critical(f"Доступный баланс {total_equity:.4f} USDT меньше требуемого {total_required:.4f} USDT")
            return None
        if quantity < MIN_QUANTITY:
            logging.critical(f"Размер позиции {quantity} ниже минимального {MIN_QUANTITY}")
            return None

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": "Buy" if signal == "buy" else "Sell",
            "orderType": "Market",
            "qty": str(quantity),
            "timeInForce": "GoodTillCancel",
            "positionIdx": 0,
            "isIsolated": True
        }
        logging.info(f"Параметры ордера: {params}, Стоимость позиции={position_value:.4f} USDT, Маржа={required_margin:.4f} USDT, Комиссия={commission:.4f} USDT, Всего требуется={total_required:.4f} USDT")
        order = api_request_with_retry(session.place_order, **params)
        if order["retCode"] == 0:
            logging.info(f"Ордер выполнен: {order}")
            return order
        else:
            logging.critical(f"Ошибка размещения ордера: {order.get('retMsg', 'Unknown error')}, retCode: {order['retCode']}")
            return None
    except Exception as e:
        logging.critical(f"Ошибка размещения ордера: {e}")
        return None

# Инициализация DataFrame историческими данными
def initialize_df():
    global df
    try:
        logging.info("Инициализация DataFrame историческими данными")
        new_data = api_request_with_retry(
            session.get_kline,
            category="linear",
            symbol=SYMBOL,
            interval="1",
            limit=MAX_DF_ROWS
        )
        if new_data["retCode"] != 0 or not new_data["result"]["list"]:
            logging.critical("Не удалось собрать исторические данные")
            return

        data = pd.DataFrame(new_data["result"]["list"], columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        data["timestamp"] = pd.to_datetime(data["timestamp"].astype(float), unit="ms")
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            data[col] = data[col].astype(float)

        # Проверка аномалий
        invalid_rows = data[(data["high"] < data["low"]) | (data["close"] <= 0) | (data["volume"] < 0)]
        if not invalid_rows.empty:
            logging.warning(f"Обнаружены аномалии в данных: {len(invalid_rows)} строк")
            data = data.drop(invalid_rows.index)

        data = data.sort_values(by="timestamp")
        data.set_index("timestamp", inplace=True)

        data["rsi"] = rsi(data["close"], length=14)
        data["atr"] = atr(data["high"], data["low"], data["close"], length=ATR_PERIOD)
        macd_df_local = macd(data["close"], fast=12, slow=26, signal=9)
        if macd_df_local is not None and all(col in macd_df_local for col in ["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"]):
            data["macd"] = macd_df_local["MACD_12_26_9"]
            data["macd_signal"] = macd_df_local["MACDs_12_26_9"]
            data["macd_hist"] = macd_df_local["MACDh_12_26_9"]
        else:
            data[["macd", "macd_signal", "macd_hist"]] = 0.0
        adx_result_local = adx(data["high"], data["low"], data["close"], length=14)
        data["adx"] = adx_result_local["ADX_14"] if adx_result_local is not None and "ADX_14" in adx_result_local else 0.0
        data["ema_fast"] = data["close"].ewm(span=20, adjust=False).mean()
        data["ema_slow"] = data["close"].ewm(span=50, adjust=False).mean()
        data["ema_diff"] = data["ema_fast"] - data["ema_slow"]
        data["trend"] = np.where(data["ema_fast"] > data["ema_slow"], 1, -1)
        data["volume_sma"] = data["volume"].rolling(window=20).mean()

        data.reset_index(inplace=True)
        data = data.dropna(subset=["open", "high", "low", "close", "volume", "turnover"])
        data[["rsi", "atr", "macd", "macd_signal", "macd_hist", "adx", "ema_diff", "ema_fast", "ema_slow", "volume_sma"]] = data[["rsi", "atr", "macd", "macd_signal", "macd_hist", "adx", "ema_diff", "ema_fast", "ema_slow", "volume_sma"]].fillna(0)
        data["trend"] = data["trend"].fillna(0).astype(int)
        df = data[df.columns]
        logging.info(f"DataFrame инициализирован с {len(df)} строками")
    except Exception as e:
        logging.critical(f"Ошибка инициализации DataFrame: {e}")

# Дообучение модели
def retrain_model():
    global model, last_retrain_time, retrain_running, scaler
    if retrain_running:
        logging.info("Дообучение уже выполняется, пропускаем")
        return
    retrain_running = True
    try:
        logging.info("Начало дообучения модели")
        new_data = api_request_with_retry(
            session.get_kline,
            category="linear",
            symbol=SYMBOL,
            interval="1",
            limit=5000  # Увеличено с 2000 до 5000
        )
        if new_data["retCode"] != 0 or not new_data["result"]["list"]:
            logging.critical("Не удалось собрать данные для дообучения")
            return

        data = pd.DataFrame(new_data["result"]["list"], columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        data["timestamp"] = pd.to_datetime(data["timestamp"].astype(float), unit="ms")
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            data[col] = data[col].astype(float)

        # Проверка аномалий
        invalid_rows = data[(data["high"] < data["low"]) | (data["close"] <= 0) | (data["volume"] < 0)]
        if not invalid_rows.empty:
            logging.warning(f"Обнаружены аномалии в данных для дообучения: {len(invalid_rows)} строк")
            data = data.drop(invalid_rows.index)

        data = data.sort_values(by="timestamp")
        data.set_index("timestamp", inplace=True)

        data["rsi"] = rsi(data["close"], length=14)
        data["atr"] = atr(data["high"], data["low"], data["close"], length=ATR_PERIOD)
        macd_df_local = macd(data["close"], fast=12, slow=26, signal=9)
        if macd_df_local is None or not all(col in macd_df_local for col in ["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"]):
            logging.critical("MACD вернул None или некорректный DataFrame в retrain_model")
            return
        data["macd"] = macd_df_local["MACD_12_26_9"]
        data["macd_signal"] = macd_df_local["MACDs_12_26_9"]
        data["macd_hist"] = macd_df_local["MACDh_12_26_9"]
        adx_result_local = adx(data["high"], data["low"], data["close"], length=14)
        data["adx"] = adx_result_local["ADX_14"] if adx_result_local is not None and "ADX_14" in adx_result_local else 0.0
        data["ema_fast"] = data["close"].ewm(span=20, adjust=False).mean()
        data["ema_slow"] = data["close"].ewm(span=50, adjust=False).mean()
        data["ema_diff"] = data["ema_fast"] - data["ema_slow"]
        data["trend"] = np.where(data["ema_fast"] > data["ema_slow"], 1, -1)
        data["volume_sma"] = data["volume"].rolling(window=20).mean()

        data.reset_index(inplace=True)
        data["future_return"] = data["close"].shift(-5) / data["close"] - 1
        data["signal"] = np.where(data["future_return"] > 0.003, 1, np.where(data["future_return"] < -0.003, -1, 0))
        data = data.dropna(subset=["open", "high", "low", "close", "volume", "turnover"])
        data[["rsi", "atr", "macd", "macd_signal", "macd_hist", "adx", "ema_diff", "ema_fast", "ema_slow", "volume_sma"]] = data[["rsi", "atr", "macd", "macd_signal", "macd_hist", "adx", "ema_diff", "ema_fast", "ema_slow", "volume_sma"]].fillna(0)
        data["trend"] = data["trend"].fillna(0).astype(int)
        logging.info(f"После обработки и удаления NaN осталось {len(data)} строк")

        if len(data) < time_steps:
            logging.critical("Недостаточно данных для дообучения")
            return

        X = data[features].values
        y = data["signal"].values + 1
        logging.info(f"X shape: {X.shape}, y shape: {y.shape}, Количество признаков: {len(features)}")

        if X.shape[1] != len(features):
            logging.critical(f"Несоответствие количества признаков: получено {X.shape[1]}, ожидается {len(features)}")
            return

        scaler.fit(X)
        joblib.dump(scaler, "scaler.pkl")
        logging.info("Scaler переобучен и сохранен")

        X_scaled = scaler.transform(X)
        X_seq, y_seq = create_sequences(X_scaled, y, time_steps)
        logging.info(f"X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")

        if len(X_seq) > 0:
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            logging.info("Модель перекомпилирована с новым оптимизатором")
            model.fit(X_seq, y_seq, epochs=10, batch_size=8, verbose=0)
            model.save("trading_model.h5", overwrite=True, save_format="h5")
            last_retrain_time = time.time()
            logging.info("Модель дообучена и сохранена")
        else:
            logging.critical("Нет данных для дообучения после создания последовательностей")
    except Exception as e:
        logging.critical(f"Ошибка при дообучении: {e}")
    finally:
        retrain_running = False
        gc.collect()

# Создание последовательностей
def create_sequences(X, y=None, time_steps=20):
    try:
        logging.info(f"Создание последовательностей: X shape={X.shape}, y={y is not None}, time_steps={time_steps}")
        Xs = []
        if y is not None:
            ys = []
            for i in range(len(X) - time_steps):
                Xs.append(X[i:i + time_steps])
                ys.append(y[i + time_steps])
            Xs = np.array(Xs)
            ys = np.array(ys)
            logging.info(f"Создано последовательностей: Xs shape={Xs.shape}, ys shape={ys.shape}")
            return Xs, ys
        else:
            for i in range(len(X) - time_steps, len(X)):
                Xs.append(X[i - time_steps:i])
            Xs = np.array(Xs)
            logging.info(f"Создано последовательностей: Xs shape={Xs.shape}")
            return Xs
    except Exception as e:
        logging.critical(f"Ошибка в create_sequences: {e}")
        return np.array([])

# Логирование статистики
def log_trade_stats():
    win_rate = successful_trades / total_trades if total_trades > 0 else 0
    logging.info(f"Статистика: Успешных сделок={successful_trades}, Всего сделок={total_trades}, Win Rate={win_rate:.2%}, Loss Streak={loss_streak}")

# Диагностика торговых сигналов
def diagnose_trading_signals(prediction, rsi_value, adx_value, trend, latest_atr):
    """Детальная диагностика почему сигналы отклоняются"""
    buy_prob = prediction[0][2]
    sell_prob = prediction[0][0]
    hold_prob = prediction[0][1]
    
    logging.warning(f"=== ДИАГНОСТИКА СИГНАЛОВ ===")
    logging.warning(f"Вероятности: Buy={buy_prob:.3f}, Sell={sell_prob:.3f}, Hold={hold_prob:.3f}")
    logging.warning(f"Индикаторы: RSI={rsi_value:.2f}, ADX={adx_value:.2f}, Trend={trend}, ATR={latest_atr:.5f}")
    
    # Проверка условий для Buy (убрали ADX)
    buy_conditions = {
        "probability": buy_prob > 0.25,
        "rsi_range": 25 < rsi_value < 75,
        "trend": trend == 1
    }
    
    # Проверка условий для Sell (убрали ADX)
    sell_conditions = {
        "probability": sell_prob > 0.25,
        "rsi_range": 25 < rsi_value < 75,
        "trend": trend == -1
    }
    
    logging.warning(f"Условия для BUY: {buy_conditions}")
    logging.warning(f"Условия для SELL: {sell_conditions}")
    
    # Рекомендации
    if not any(buy_conditions.values()) and not any(sell_conditions.values()):
        logging.warning("РЕКОМЕНДАЦИЯ: Все условия не выполнены - HOLD")
    elif buy_prob > 0.25 and trend != 1:
        logging.warning("РЕКОМЕНДАЦИЯ: Для BUY требуется подтверждение тренда")
    elif sell_prob > 0.25 and trend != -1:
        logging.warning("РЕКОМЕНДАЦИЯ: Для SELL требуется подтверждение тренда")
    
    logging.warning(f"=== КОНЕЦ ДИАГНОСТИКИ ===")

# Обработка свечей
def handle_message(message):
    global df, position_type, entry_price, take_profit_price, stop_loss_price, successful_trades, total_trades, last_trade_time, last_message_time, loss_streak, pause_until, data_buffer
    try:
        current_time = time.time()
        data_buffer.append(message)
        if len(data_buffer) < 3:  # Уменьшаем с 5 до 3 для более быстрой обработки
            return
        messages = data_buffer.copy()
        data_buffer.clear()

        last_message_time = current_time
        logging.info(f"Обработка {len(messages)} сообщений из буфера")

        for message in messages:
            if "data" not in message or not message["data"]:
                continue

            new_rows = []
            for data in message["data"]:
                if not all(key in data for key in ["start", "open", "high", "low", "close", "volume", "turnover"]):
                    logging.critical(f"Некорректные данные в сообщении: {list(data.keys())}")
                    continue

                # Проверка аномалий
                if float(data["high"]) < float(data["low"]) or float(data["close"]) <= 0 or float(data["volume"]) < 0:
                    logging.warning(f"Аномалия в свече: timestamp={data['start']}, high={data['high']}, low={data['low']}, close={data['close']}, volume={data['volume']}")
                    continue

                # Логируем только важные свечи
                if float(data['volume']) > 1000000:  # Только при большом объеме
                    logging.critical(f"Высокий объем: {data['close']}, vol={data['volume']}")

                new_row = {
                    "timestamp": pd.to_datetime(int(data["start"]), unit="ms"),
                    "open": float(data["open"]),
                    "high": float(data["high"]),
                    "low": float(data["low"]),
                    "close": float(data["close"]),
                    "volume": float(data["volume"]),
                    "turnover": float(data["turnover"]),
                    "rsi": 0.0,
                    "atr": 0.0,
                    "macd": 0.0,
                    "macd_signal": 0.0,
                    "macd_hist": 0.0,
                    "adx": 0.0,
                    "ema_diff": 0.0,
                    "trend": 0,
                    "ema_fast": 0.0,
                    "ema_slow": 0.0,
                    "volume_sma": 0.0
                }
                new_rows.append(new_row)

            if not new_rows:
                continue

            # Важно: не зависеть от текущих колонок df при создании new_df
            new_df = pd.DataFrame(new_rows)
            new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])

            df = ensure_timestamp_column(df)
            logging.info(f"Тип индекса df: {type(df.index)}, Тип индекса new_df: {type(new_df.index)}")

            # Конкатенация и нормализация структуры
            df = pd.concat([df, new_df], ignore_index=True)
            df = ensure_timestamp_column(df)
            df = df.sort_values(by="timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
            if len(df) > MAX_DF_ROWS:
                df.drop(df.index[:len(df) - MAX_DF_ROWS], inplace=True)
                df.reset_index(drop=True, inplace=True)

            # Логируем только при изменении количества свечей
            if len(df) % 50 == 0:
                logging.critical(f"Собрано свечей: {len(df)}")

            # Проверка баланса
            balance_response = api_request_with_retry(session.get_wallet_balance, accountType="UNIFIED")
            if balance_response["retCode"] != 0:
                logging.critical(f"Ошибка получения баланса: {balance_response.get('retMsg', 'Unknown error')}, retCode: {balance_response['retCode']}")
                return
            try:
                balance_data = balance_response["result"]["list"][0]
                logging.info(f"Полный ответ баланса: {balance_data}")
                total_equity = float(balance_data.get("totalEquity", 0))
                logging.info(f"Текущий баланс: TotalEquity={total_equity:.4f} USDT")
                if total_equity < FIXED_BALANCE:
                    logging.critical(f"Недостаточный баланс: {total_equity:.4f} USDT, требуется минимум {FIXED_BALANCE} USDT")
                    return
            except (IndexError, KeyError, ValueError) as e:
                logging.critical(f"Ошибка обработки ответа баланса: {e}, Полный ответ: {balance_response}")
                return

            # Проверка статуса позиции
            position_info = api_request_with_retry(session.get_positions, category="linear", symbol=SYMBOL)
            if position_info["retCode"] == 0 and position_info["result"]["list"]:
                for pos in position_info["result"]["list"]:
                    logging.info(f"Позиция: size={pos['size']}, side={pos['side']}")
                    if float(pos["size"]) == 0 and position_type is not None:
                        logging.info(f"Позиция закрыта (size=0), сбрасываем position_type: {position_type} -> None")
                        try:
                            order_history = api_request_with_retry(
                                session.get_closed_pnl,
                                category="linear",
                                symbol=SYMBOL,
                                limit=1
                            )
                            if order_history["retCode"] == 0 and order_history["result"]["list"]:
                                closed_pnl = float(order_history["result"]["list"][0].get("closedPnl", 0))
                                if closed_pnl > 0:
                                    successful_trades += 1
                                    loss_streak = 0
                                else:
                                    loss_streak += 1
                                    if loss_streak >= 2:
                                        logging.critical(f"Достигнуто {loss_streak} убыточных сделки подряд, пауза на {loss_streak * 2} часов")
                                        pause_until = current_time + loss_streak * 2 * 60 * 60
                                qty = float(order_history["result"]["list"][0].get("qty", MIN_QUANTITY))
                                avg_price = float(order_history["result"]["list"][0].get("avgEntryPrice", entry_price))
                                commission = qty * avg_price * TAKER_FEE * 2
                                logging.info(f"Позиция закрыта: P&L={closed_pnl:.4f} USDT, Commission={commission:.4f} USDT, Net Profit={closed_pnl - commission:.4f} USDT")
                                log_trade_stats()
                        except Exception as e:
                            logging.critical(f"Ошибка при получении P&L: {e}")
                        position_type = None
                        entry_price = 0.0
                        take_profit_price = 0.0
                        stop_loss_price = 0.0
            else:
                logging.critical(f"Ошибка проверки позиции: {position_info.get('retMsg', 'Unknown error')}")

            # Проверка паузы из-за убытков
            if current_time < pause_until:
                logging.info(f"Торговля приостановлена до {pd.to_datetime(pause_until, unit='s')}, собираем данные")
                continue

            # Проверка данных
            if len(df) < time_steps:
                logging.critical(f"Недостаточно свечей для обработки: {len(df)} < {time_steps}")
                continue
            
            # Начинаем торговую логику

            required_columns = ["open", "high", "low", "close", "volume", "turnover"]
            if not all(col in df.columns for col in required_columns + ["timestamp"]):
                logging.critical(f"Отсутствуют необходимые столбцы в df: {df.columns}")
                df = ensure_timestamp_column(df)
                continue
            if df[required_columns].isna().any().any():
                logging.critical(f"Обнаружены NaN в данных: {df[required_columns].isna().sum()}")
                continue

            latest_data_row = df[required_columns].iloc[-1]
            logging.info(f"Последние данные: open={latest_data_row['open']:.5f}, high={latest_data_row['high']:.5f}, "
                         f"low={latest_data_row['low']:.5f}, close={latest_data_row['close']:.5f}, "
                         f"volume={latest_data_row['volume']:.2f}, turnover={latest_data_row['turnover']:.2f}")
            if latest_data_row["high"] < latest_data_row["low"]:
                logging.critical(f"Аномалия в данных: high ({latest_data_row['high']:.5f}) < low ({latest_data_row['low']:.5f})")
                continue
            if latest_data_row["high"] == latest_data_row["low"] and latest_data_row["volume"] > 0:
                logging.warning(f"Потенциальная аномалия: high == low ({latest_data_row['high']:.5f}), но volume > 0")
            if df["close"].isna().any():
                logging.critical(f"Обнаружены NaN в close: {df['close'].isna().sum()}")
                continue
            if df["close"].nunique() <= 1:
                logging.warning("Все значения close одинаковы, пропускаем обработку")
                continue

            # Вычисление индикаторов (безопасно)
            min_rows = max(ATR_PERIOD, 14, 20)
            if len(df) >= min_rows:
                df = update_indicators_safely(df)
                try:
                    logging.critical(f"Индикаторы рассчитаны: RSI={df['rsi'].iloc[-1]:.2f}, ATR={df['atr'].iloc[-1]:.4f}, ADX={df['adx'].iloc[-1]:.2f}")
                    logging.info(f"Индикаторы: RSI={df['rsi'].iloc[-1]:.2f}, ATR={df['atr'].iloc[-1]:.4f}, "
                                 f"MACD={df['macd'].iloc[-1]:.4f}, ADX={df['adx'].iloc[-1]:.2f}, "
                                 f"Trend={df['trend'].iloc[-1]}")
                except Exception:
                    pass
            else:
                logging.info(f"Недостаточно свечей для индикаторов: {len(df)} < {min_rows}")
                continue

            # Проверка волатильности
            latest_atr = df["atr"].iloc[-1]
            if pd.isna(latest_atr):
                logging.critical(f"ATR является NaN, используем MIN_ATR={MIN_ATR}")
                latest_atr = MIN_ATR
            if latest_atr < MIN_ATR:
                logging.critical(f"Волатильность слишком низкая: ATR={latest_atr:.4f} < {MIN_ATR}")
                continue
            
            # Проверка объема (сильно снижен порог)
            if df["volume"].iloc[-1] < df["volume_sma"].iloc[-1] * 0.3:  # Снижен до 0.3 (30% от среднего)
                logging.critical(f"Объем слишком низкий: {df['volume'].iloc[-1]:.2f} < {df['volume_sma'].iloc[-1] * 0.3:.2f}")
                continue
            
            # --- Soft risk controls (опционально) ---
            if position_type is not None:
                latest_close_soft = df["close"].iloc[-1]
                # Катастрофический выход по ATR (если включен)
                if SOFT_RISK_CUTOFF_ATR and SOFT_RISK_CUTOFF_ATR > 0:
                    adverse_ok = False
                    if position_type == "long" and (entry_price - latest_close_soft) >= SOFT_RISK_CUTOFF_ATR * latest_atr:
                        adverse_ok = True
                    if position_type == "short" and (latest_close_soft - entry_price) >= SOFT_RISK_CUTOFF_ATR * latest_atr:
                        adverse_ok = True
                    if adverse_ok:
                        logging.critical(f"Аварийный выход по ATR: entry={entry_price:.5f}, last={latest_close_soft:.5f}, ATR={latest_atr:.5f}")
                        try:
                            qty_to_close = MIN_QUANTITY
                            position_info_soft = api_request_with_retry(session.get_positions, category="linear", symbol=SYMBOL)
                            if position_info_soft["retCode"] == 0 and position_info_soft["result"]["list"]:
                                for pos in position_info_soft["result"]["list"]:
                                    if float(pos["size"]) > 0:
                                        qty_to_close = int(float(pos["size"]))
                                        break
                            close_params_soft = {
                                "category": "linear",
                                "symbol": SYMBOL,
                                "side": "Sell" if position_type == "long" else "Buy",
                                "orderType": "Market",
                                "qty": str(qty_to_close),
                                "positionIdx": 0,
                                "isIsolated": True
                            }
                            api_request_with_retry(session.place_order, **close_params_soft)
                            position_type = None
                            entry_price = 0.0
                            take_profit_price = 0.0
                            stop_loss_price = 0.0
                            logging.critical("Позиция закрыта аварийно по ATR")
                        except Exception as e:
                            logging.critical(f"Ошибка аварийного выхода: {e}")

                # Тайм-аут позиции (если включен)
                if MAX_HOLD_SECONDS and MAX_HOLD_SECONDS > 0 and last_trade_time > 0:
                    if time.time() - last_trade_time > MAX_HOLD_SECONDS:
                        logging.critical(f"Тайм-аут позиции: удержание > {MAX_HOLD_SECONDS}s — закрываем")
                        try:
                            qty_to_close = MIN_QUANTITY
                            position_info_soft = api_request_with_retry(session.get_positions, category="linear", symbol=SYMBOL)
                            if position_info_soft["retCode"] == 0 and position_info_soft["result"]["list"]:
                                for pos in position_info_soft["result"]["list"]:
                                    if float(pos["size"]) > 0:
                                        qty_to_close = int(float(pos["size"]))
                                        break
                            close_params_soft = {
                                "category": "linear",
                                "symbol": SYMBOL,
                                "side": "Sell" if position_type == "long" else "Buy",
                                "orderType": "Market",
                                "qty": str(qty_to_close),
                                "positionIdx": 0,
                                "isIsolated": True
                            }
                            api_request_with_retry(session.place_order, **close_params_soft)
                            position_type = None
                            entry_price = 0.0
                            take_profit_price = 0.0
                            stop_loss_price = 0.0
                            logging.critical("Позиция закрыта по тайм-ауту")
                        except Exception as e:
                            logging.critical(f"Ошибка закрытия по тайм-ауту: {e}")

            # Проверка рыночного режима (ADX отключен)
            adx_value = df["adx"].iloc[-1]

            latest_data = df[features].iloc[-time_steps:].values
            # Подготовка данных для модели
            if latest_data.shape[1] != len(features):
                logging.critical(f"Несоответствие признаков: получено {latest_data.shape[1]}, ожидается {len(features)}")
                continue
            if np.any(np.isnan(latest_data)):
                logging.critical(f"Обнаружены NaN в latest_data")
                continue
            latest_data_scaled = scaler.transform(latest_data)
            latest_data_seq = latest_data_scaled.reshape(1, time_steps, len(features))
            prediction = model.predict(latest_data_seq, batch_size=1, verbose=0)
            signal = np.argmax(prediction)
            rsi_value = df["rsi"].iloc[-1]
            trend = df["trend"].iloc[-1]

            # Подробное логирование сигналов (только при важных событиях)
            if prediction[0][2] > 0.3 or prediction[0][0] > 0.3:  # Логируем только при высоких вероятностях
                logging.warning(f"Модель предсказала: Buy={prediction[0][2]:.2f}, Hold={prediction[0][1]:.2f}, Sell={prediction[0][0]:.2f}, "
                             f"RSI={rsi_value:.2f}, ADX={adx_value:.2f}, Trend={trend}, ATR={latest_atr:.5f}")
            
            # Диагностика каждые 10 минут или при высоких вероятностях
            if (prediction[0][2] > 0.2 or prediction[0][0] > 0.2) and int(time.time()) % 600 == 0:
                diagnose_trading_signals(prediction, rsi_value, adx_value, trend, latest_atr)

            # Фильтры сигналов (сбалансированные пороги для прибыльности)
            reasons = []
            
            # Дополнительные условия для качества сигналов (снижены пороги)
            buy_conditions = (
                prediction[0][2] > 0.15 and  # Снижено с 20% до 15%
                30 < rsi_value < 70 and      # RSI в разумном диапазоне
                trend == 1 and               # Бычий тренд
                latest_atr > MIN_ATR * 1.5   # Достаточная волатильность
            )
            
            sell_conditions = (
                prediction[0][0] > 0.15 and  # Снижено с 20% до 15%
                30 < rsi_value < 70 and      # RSI в разумном диапазоне
                trend == -1 and              # Медвежий тренд
                latest_atr > MIN_ATR * 1.5   # Достаточная волатильность
            )
            
            if buy_conditions:
                signal = 2
                logging.warning(f"BUY сигнал: prob={prediction[0][2]:.2f}, RSI={rsi_value:.1f}, trend={trend}, ATR={latest_atr:.4f}")
            elif sell_conditions:
                signal = 0
                logging.warning(f"SELL сигнал: prob={prediction[0][0]:.2f}, RSI={rsi_value:.1f}, trend={trend}, ATR={latest_atr:.4f}")
            else:
                signal = 1
                if prediction[0][2] <= 0.2:
                    reasons.append(f"Buy probability low: {prediction[0][2]:.2f} <= 0.2")
                if prediction[0][0] <= 0.2:
                    reasons.append(f"Sell probability low: {prediction[0][0]:.2f} <= 0.2")
                if rsi_value >= 70:
                    reasons.append(f"RSI too high: {rsi_value:.1f} >= 70")
                if rsi_value <= 30:
                    reasons.append(f"RSI too low: {rsi_value:.1f} <= 30")
                if trend != 1:
                    reasons.append(f"Trend not bullish: {trend}")
                if trend != -1:
                    reasons.append(f"Trend not bearish: {trend}")
                if latest_atr <= MIN_ATR * 1.5:
                    reasons.append(f"ATR too low: {latest_atr:.4f} <= {MIN_ATR * 1.5:.4f}")
            
            logging.warning(f"Сигнал после фильтров: {signal}, Причины: {', '.join(reasons)}")
            
            # Дополнительная диагностика при отклонении сигналов
            if signal == 1 and (prediction[0][2] > 0.2 or prediction[0][0] > 0.2):
                logging.warning(f"СИГНАЛ ОТКЛОНЕН: Модель предлагала торговлю, но фильтры заблокировали")
                logging.warning(f"Детали: Buy_prob={prediction[0][2]:.3f}, Sell_prob={prediction[0][0]:.3f}")
                logging.warning(f"Фильтры: ADX={adx_value:.2f} (мин=20), RSI={rsi_value:.2f} (диапазон=25-75), Trend={trend}")

            latest_close = df["close"].iloc[-1]
            if current_time - last_trade_time < MIN_TRADE_INTERVAL:
                logging.info(f"Торговля заблокирована: интервал между сделками {MIN_TRADE_INTERVAL} секунд")
                continue

            logging.warning(f"Проверка условий торговли: signal={signal}, position_type={position_type}")
            if signal == 2 and position_type is None:
                # Дополнительная проверка качества сигнала
                if prediction[0][2] > 0.18 and rsi_value > 35 and rsi_value < 65:
                    logging.warning("Условие для Buy выполнено, пытаемся открыть позицию")
                    order = place_order("buy", current_price=latest_close)
                else:
                    logging.warning(f"Buy сигнал отклонен: prob={prediction[0][2]:.2f}, RSI={rsi_value:.1f} (требуется prob>0.18, RSI 35-65)")
                    order = None
                if order:
                    position_type = "long"
                    entry_price = latest_close
                    take_profit_price = entry_price + max(ATR_MULTIPLIER_TP * latest_atr, entry_price * MIN_TP_DISTANCE)
                    stop_loss_price = entry_price - max(ATR_MULTIPLIER_SL * latest_atr, entry_price * MIN_SL_DISTANCE)
                    quantity = int(order["result"].get("orderQty", MIN_QUANTITY))
                    logging.info(f"Расчет для лонга: QUANTITY={quantity}, Entry={entry_price:.5f}, TP={take_profit_price:.5f}, SL={stop_loss_price:.5f}")
                    if set_trading_stop(position_type, entry_price, take_profit_price, stop_loss_price, quantity):
                        logging.info(f"Открыта длинная позиция: Entry={entry_price:.5f}, TP={take_profit_price:.5f}, SL={'OFF' if not USE_HARD_SL else f'{stop_loss_price:.5f}'}")
                        total_trades += 1
                        last_trade_time = current_time
                    else:
                        logging.critical("Отмена позиции из-за невозможности установить TP/SL")
                        close_params = {
                            "category": "linear",
                            "symbol": SYMBOL,
                            "side": "Sell",
                            "orderType": "Market",
                            "qty": str(quantity),
                            "positionIdx": 0,
                            "isIsolated": True
                        }
                        api_request_with_retry(session.place_order, **close_params)
                        position_type = None
            elif signal == 0 and position_type is None:
                # Дополнительная проверка качества сигнала
                # Для SELL: RSI может быть низким (перепроданность) или высоким (перекупленность)
                if prediction[0][0] > 0.18 and (rsi_value < 35 or rsi_value > 65):
                    logging.warning("Условие для Sell выполнено, пытаемся открыть позицию")
                    order = place_order("sell", current_price=latest_close)
                else:
                    logging.warning(f"Sell сигнал отклонен: prob={prediction[0][0]:.2f}, RSI={rsi_value:.1f} (требуется prob>0.18, RSI<35 или RSI>65)")
                    order = None
                if order:
                    position_type = "short"
                    entry_price = latest_close
                    take_profit_price = entry_price - max(ATR_MULTIPLIER_TP * latest_atr, entry_price * MIN_TP_DISTANCE)
                    stop_loss_price = entry_price + max(ATR_MULTIPLIER_SL * latest_atr, entry_price * MIN_SL_DISTANCE)
                    quantity = int(order["result"].get("orderQty", MIN_QUANTITY))
                    logging.info(f"Расчет для шорта: QUANTITY={quantity}, Entry={entry_price:.5f}, TP={take_profit_price:.5f}, SL={stop_loss_price:.5f}")
                    if set_trading_stop(position_type, entry_price, take_profit_price, stop_loss_price, quantity):
                        logging.info(f"Открыта короткая позиция: Entry={entry_price:.5f}, TP={take_profit_price:.5f}, SL={'OFF' if not USE_HARD_SL else f'{stop_loss_price:.5f}'}")
                        total_trades += 1
                        last_trade_time = current_time
                    else:
                        logging.critical("Отмена позиции из-за невозможности установить TP/SL")
                        close_params = {
                            "category": "linear",
                            "symbol": SYMBOL,
                            "side": "Buy",
                            "orderType": "Market",
                            "qty": str(quantity),
                            "positionIdx": 0,
                            "isIsolated": True
                        }
                        api_request_with_retry(session.place_order, **close_params)
                        position_type = None
            else:
                logging.info(f"Торговля не выполняется: signal={signal}, position_type={position_type}")

            if current_time - last_retrain_time > RETRAIN_INTERVAL and not retrain_running:
                threading.Thread(target=retrain_model).start()

            gc.collect()

    except Exception as e:
        logging.critical(f"Ошибка в handle_message: {e}, Сообщение: {message}")

# Подключение к WebSocket с бесконечными попытками
def connect_websocket_with_retry(max_attempts=20, delay=15):
    global last_message_time
    attempt = 0
    while True:
        try:
            ws = WebSocket(
                testnet=False,
                api_key=API_KEY,
                api_secret=API_SECRET,
                channel_type="linear",
                ping_interval=30,
                ping_timeout=20
            )
            # Пробуем разные способы подключения
            try:
                ws.kline_stream(interval=1, symbol=SYMBOL, callback=handle_message)
                logging.critical("Подключение через kline_stream успешно")
            except Exception as e:
                logging.critical(f"Ошибка kline_stream: {e}")
                # Пробуем альтернативный способ
                ws.subscribe_kline_stream(interval=1, symbol=SYMBOL, callback=handle_message)
                logging.critical("Подключение через subscribe_kline_stream")
            logging.critical("WebSocket подключен")
            last_message_time = time.time()
            
            # Добавляем отладочную информацию
            logging.critical(f"Подписка на поток: {SYMBOL}, интервал: 1 минута")
            logging.critical("Ожидаем данные от WebSocket...")
            
            return ws
        except Exception as e:
            attempt += 1
            logging.critical(f"Попытка {attempt} неудачна: {e}")
            if attempt >= max_attempts:
                logging.critical(f"Достигнуто максимальное количество попыток ({max_attempts}), пауза перед повтором")
                attempt = 0
                time.sleep(60)
            else:
                time.sleep(delay)

# Проверка активности WebSocket
def check_websocket_activity(ws):
    global last_message_time
    while True:
        current_time = time.time()
        if current_time - last_message_time > 30:
            logging.critical("WebSocket неактивен более 30 секунд, пытаемся переподключиться")
            try:
                ws.close()
            except:
                pass
            ws = connect_websocket_with_retry()
        time.sleep(15)

# Запуск бота
if __name__ == "__main__":
    logging.info("Инициализация DataFrame")
    initialize_df()
    
    # Проверяем получение данных через REST API
    logging.critical("Проверка получения данных через REST API...")
    test_data = api_request_with_retry(
        session.get_kline,
        category="linear",
        symbol=SYMBOL,
        interval="1",
        limit=5
    )
    if test_data["retCode"] == 0:
        logging.critical(f"REST API работает: получено {len(test_data['result']['list'])} свечей")
    else:
        logging.critical(f"Ошибка REST API: {test_data.get('retMsg', 'Unknown')}")
    
    logging.info("Запуск дообучения модели при старте")
    threading.Thread(target=retrain_model).start()
    
    ws = connect_websocket_with_retry()
    threading.Thread(target=check_websocket_activity, args=(ws,), daemon=True).start()
    try:
        while True:
            # Проверяем активность WebSocket каждые 30 секунд
            if int(time.time()) % 30 == 0:
                logging.critical(f"Проверка активности: время с последнего сообщения = {time.time() - last_message_time:.1f} сек")
            time.sleep(1)
    except KeyboardInterrupt:
        logging.critical("Скрипт остановлен пользователем")
    except Exception as e:
        logging.critical(f"Ошибка в главном цикле: {e}")

