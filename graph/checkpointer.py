"""
Day 25 — SQLite Checkpointing

Her LangGraph run'ının state'ini SQLite'a persist eder.
Crash sonrası aynı thread_id ile resume edilebilir.

Kullanım:
    from graph.checkpointer import make_checkpointer

    with make_checkpointer() as checkpointer:
        graph = build_finance_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "run-NVDA-001"}}

        # İlk run
        result = graph.invoke({"ticker": "NVDA", "messages": []}, config)

        # Crash sonrası aynı thread_id → kaldığı yerden devam
        state = graph.get_state(config)
        print(state.values)

Neden thread_id?
    Her run'ı birbirinden izole eder. NVDA ve TSLA run'ları aynı DB'de
    birbirine karışmaz. Debug için belirli bir run'ın checkpoint'ini
    çekip inceleyebilirsin.
"""

from contextlib import contextmanager

from langgraph.checkpoint.sqlite import SqliteSaver


@contextmanager
def make_checkpointer(db_path: str = "checkpoints.db"):
    """
    SqliteSaver context manager. `with` bloğu dışına çıkınca DB bağlantısı kapanır.

        with make_checkpointer() as cp:
            graph = build_finance_graph(checkpointer=cp)
            ...

    db_path: SQLite dosyasının yolu. Varsayılan proje kökünde checkpoints.db.
    """
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        yield checkpointer
