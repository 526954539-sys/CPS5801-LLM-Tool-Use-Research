import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "data/raw/DAMO-ConvAI/api-bank"
sys.path.insert(0, str(API_ROOT))

from apis.check_token import CheckToken
from apis.get_user_token import GetUserToken
from apis.add_agenda import AddAgenda
from apis.add_alarm import AddAlarm
from apis.query_alarm import QueryAlarm

ACCOUNT_FIXTURE = {
    "sandbox_user": {
        "password": "sandbox_password_only",
        "token": "sandbox_token_only",
    }
}
AGENDA_FIXTURE = {}
ALARM_FIXTURE = {}

def new_sandbox():
    account = copy.deepcopy(ACCOUNT_FIXTURE)
    agenda = copy.deepcopy(AGENDA_FIXTURE)
    alarm = copy.deepcopy(ALARM_FIXTURE)

    checker = CheckToken(account)
    login = GetUserToken(account)
    add_agenda = AddAgenda(agenda, checker)
    add_alarm = AddAlarm(alarm, checker)
    query_alarm = QueryAlarm(alarm, checker)

    return {
        "account": account,
        "agenda": agenda,
        "alarm": alarm,
        "checker": checker,
        "login": login,
        "add_agenda": add_agenda,
        "add_alarm": add_alarm,
        "query_alarm": query_alarm,
    }

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def run():
    original_account = copy.deepcopy(ACCOUNT_FIXTURE)
    original_agenda = copy.deepcopy(AGENDA_FIXTURE)
    original_alarm = copy.deepcopy(ALARM_FIXTURE)

    box = new_sandbox()

    login = box["login"].call("sandbox_user", "sandbox_password_only")
    require(login["exception"] is None, "虚拟账户登录失败")
    require(login["output"]["token"] == "sandbox_token_only", "登录 token 不匹配")

    checked = box["checker"].call("sandbox_token_only")
    require(checked["exception"] is None, "虚拟 token 校验失败")
    require(checked["output"]["username"] == "sandbox_user", "用户名映射错误")

    bad_login = box["login"].call("sandbox_user", "wrong_password")
    require(bad_login["exception"] is not None, "错误密码未被拒绝")

    agenda_result = box["add_agenda"].call(
        "sandbox_token_only",
        "Sandbox meeting",
        "2026-12-01 14:00:00",
        "Sandbox room",
    )
    require(agenda_result["exception"] is None, "添加虚拟日程失败")
    require(len(box["agenda"]) == 1, "Agenda 内存状态未正确更新")

    alarm_result = box["add_alarm"].call(
        "sandbox_token_only",
        "2026-12-01 13:50:00",
    )
    require(alarm_result["exception"] is None, "添加虚拟闹钟失败")
    require(len(box["alarm"]) == 1, "Alarm 内存状态未正确更新")

    query_result = box["query_alarm"].call(
        "sandbox_token_only",
        "2026-12-01 13:50:00",
    )
    require(query_result["exception"] is None, "查询刚添加的虚拟闹钟失败")
    require(query_result["output"]["username"] == "sandbox_user", "查询用户不匹配")

    bad_token = box["query_alarm"].call(
        "invalid_sandbox_token",
        "2026-12-01 13:50:00",
    )
    require(bad_token["exception"] is not None, "无效 token 未被拒绝")

    fresh_box = new_sandbox()
    require(len(fresh_box["agenda"]) == 0, "新任务继承了旧 Agenda 状态")
    require(len(fresh_box["alarm"]) == 0, "新任务继承了旧 Alarm 状态")

    require(ACCOUNT_FIXTURE == original_account, "账户 fixture 被修改")
    require(AGENDA_FIXTURE == original_agenda, "Agenda fixture 被修改")
    require(ALARM_FIXTURE == original_alarm, "Alarm fixture 被修改")

    print("PASS: 虚拟账户登录与 token 校验")
    print("PASS: 错误密码与无效 token 被拒绝")
    print("PASS: AddAgenda 内存写入")
    print("PASS: AddAlarm 内存写入")
    print("PASS: QueryAlarm 读取共享的任务内存状态")
    print("PASS: 新任务状态重置")
    print("PASS: 原始 fixture 未被修改")
    print("SAFETY: 未加载 init_database，未调用 dump_database")
    print("RESULT: ISOLATED SANDBOX VALIDATION PASSED")

if __name__ == "__main__":
    run()
