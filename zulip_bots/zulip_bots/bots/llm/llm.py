# See readme.md for instructions on running this code.

from typing import Any, Dict, List

from zulip_bots.lib import BotHandler, ExternalBotHandler
from zulip_bots.bots.llm.providers import DEFAULT_PROVIDER, SUPPORTED_PROVIDERS


class LLMHandler:
    def initialize(self, bot_handler: BotHandler) -> None:
        if not isinstance(bot_handler, ExternalBotHandler):
            raise Exception(
                "The LLM bot only supports being run externally from zulip-run-bot or a Zulip Botserver."
            )

        self.config = bot_handler.get_config_info("llm")
        self.llm_provider_name = self.config.get("default_llm_provider", DEFAULT_PROVIDER)

        print("Loaded config", self.config)

        if self.llm_provider_name not in SUPPORTED_PROVIDERS:
            raise Exception(f'llm_provider "{self.llm_provider_name}" is not supported')

        self.llm_provider = SUPPORTED_PROVIDERS[self.llm_provider_name]()
        self.llm_provider.initialize(self.config)

    def usage(self) -> str:
        return """
        A basic LLM conversational bot
        """

    def join_messages(self, messages: List[Dict[str, Any]]) -> str:
        result = []
        for message in messages:
            result.append(f'{message["sender_full_name"]} says: {message["content"]}')
        return "<SEP>".join(result)

    def handle_summarization(
        self, stream_id: int, topic: str, bot_handler: ExternalBotHandler
    ) -> str:
        message_filters = {
            "anchor": "newest",
            "num_before": "20",
            "num_after": "0",
            "apply_markdown": False,
            "narrow": [
                {"operator": "stream", "operand": stream_id},
                {"operator": "topic", "operand": topic},
            ],
        }
        result = bot_handler._client.get_messages(message_filters)
        content = self.join_messages(result["messages"])

        return self.llm_provider.summarize(content)

    def handle_message(self, message: Dict[str, str], bot_handler: BotHandler) -> None:
        assert isinstance(bot_handler, ExternalBotHandler)

        args = message["content"].strip().split(maxsplit=1)

        if len(args) == 0 or args[0] == "help":
            bot_handler.send_reply(message, self.usage())
            return

        if args[0] == "summarize":
            topic = message["subject"] if len(args) == 1 else args[1]
            response = self.handle_summarization(message["stream_id"], topic, bot_handler)
        elif args[0] == "config":
            pass

        bot_handler.send_reply(message, response)


handler_class = LLMHandler
