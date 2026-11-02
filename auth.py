"""로컬 실습용 계정 인증. 비밀번호는 모델 입력에 전달하지 않는다."""
import getpass
import hashlib
import hmac
import json
import secrets
from pathlib import Path

ITERATIONS = 260_000
ACCOUNT_IDS = tuple(f"user_{i:03}" for i in range(1, 6))

def password_hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), ITERATIONS).hex()

def ensure_accounts(root: Path):
    path = root / "data" / "accounts.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        accounts = {}
        for index, uid in enumerate(ACCOUNT_IDS, 1):
            salt = secrets.token_hex(16)
            accounts[uid] = {"salt": salt, "password_hash": password_hash("111", salt)}
        # 이미 있는 계정 파일은 덮어쓰지 않는다.
        with path.open("x", encoding="utf-8") as file:
            json.dump(accounts, file, ensure_ascii=False, indent=2)
    accounts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(accounts, dict):
        raise ValueError("계정 파일 형식이 올바르지 않습니다.")
    return accounts

def authenticate(accounts, user_id, password):
    record = accounts.get(user_id)
    if user_id not in ACCOUNT_IDS or not isinstance(record, dict):
        return False
    try:
        return hmac.compare_digest(password_hash(password, record["salt"]), record["password_hash"])
    except (ValueError, KeyError, TypeError):
        return False

def login(root: Path, suggested_user=None):
    accounts = ensure_accounts(root)
    print("\n=== 미쯔 도서관 로그인 ===\n종료하려면 ID에 /quit를 입력하세요.")
    for attempt in range(3):
        label = f"ID [{suggested_user}] > " if suggested_user else "ID > "
        user_id = input(label).strip() or suggested_user
        if user_id == "/quit":
            return None
        password = getpass.getpass("비밀번호 > ")
        if authenticate(accounts, user_id, password):
            return user_id
        print(f"ID 또는 비밀번호가 올바르지 않습니다. ({attempt + 1}/3)")
    print("로그인에 3회 실패하여 종료합니다.")
    return None
