import logging
import asyncio
import json
from pathlib import Path
from utils.routing import safe_call
from modules.async_training_engine import AsyncTrainer
from modules.blockchain_utils import (
    update_commodity_data,
    validate_token,
    adjust_price,
)
from nebula.modules import DataManager
from nebula.train import NASManager, EvoOptimizer
from nebula.rag import LocalRAG


class Orchestrator:
    def __init__(self):
        self.modules = {}
        self.logger = logging.getLogger("Orchestrator")
        self.async_engine = AsyncTrainer(self)
        self.register("async_training", self.async_engine)
        # Expose blockchain utility functions
        self.register("update_commodity", update_commodity_data)
        self.register("validate_token", validate_token)
        self.register("adjust_price", adjust_price)
        # Register core components
        self.register("DataManager", DataManager({}))
        self.register("NAS", NASManager())
        self.register("EvoOptimizer", EvoOptimizer())
        self.register("RAG", LocalRAG())

    def register(self, name, func):
        self.modules[name] = func
        try:
            agents_file = Path('agents.json')
            data = {}
            if agents_file.exists():
                data = json.loads(agents_file.read_text())
            data[name] = getattr(func, '__class__', type(func)).__name__
            agents_file.write_text(json.dumps(data, indent=2))
            with open('agent_status.log', 'a') as logf:
                logf.write(f"Registered {name}\n")
        except Exception as e:
            self.logger.error(f"Failed to log registration for {name}: {e}")

    def call(self, name, *args, **kwargs):
        func = self.modules.get(name)
        if not func:
            self.logger.warning(f"Module {name} missing")
            return None
        result = asyncio.run(safe_call(func, *args, **kwargs))
        if result is None:
            self.logger.error(f"Call to {name} failed")
        return result

    async def call_async(self, name, *args, **kwargs):
        func = self.modules.get(name)
        if not func:
            self.logger.warning(f"Module {name} missing")
            return None
        return await safe_call(func, *args, **kwargs)
