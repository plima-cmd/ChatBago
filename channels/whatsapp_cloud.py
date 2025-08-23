# channels/whatsapp_cloud.py
import os
import json
import logging
import aiohttp
from typing import Text, Dict, Any, Optional, List

from sanic import Blueprint, response
from sanic.request import Request

from rasa.core.channels.channel import InputChannel, OutputChannel, UserMessage, CollectingOutputChannel

logger = logging.getLogger(__name__)

class WhatsAppCloudOutput(OutputChannel):
    """Envía mensajes a WhatsApp Cloud API"""

    @classmethod
    def name(cls) -> Text:
        return "whatsapp_cloud"

    def __init__(self, token: Text, phone_number_id: Text):
        self.token = token
        self.phone_number_id = phone_number_id
        self.api_url = f"https://graph.facebook.com/v20.0/{self.phone_number_id}/messages"

    async def _post(self, payload: Dict[str, Any]) -> None:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as r:
                if r.status >= 400:
                    err = await r.text()
                    logger.error(f"WhatsApp send fail {r.status}: {err}")

    async def send_text_message(self, recipient_id: Text, text: Text, **kwargs: Any) -> None:
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient_id,
            "type": "text",
            "text": {"body": text}
        }
        await self._post(payload)

    async def send_image_url(self, recipient_id: Text, image: Text, **kwargs: Any) -> None:
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient_id,
            "type": "image",
            "image": {"link": image}
        }
        await self._post(payload)

    async def send_buttons(self, recipient_id: Text, text: Text, buttons: List[Dict[str, Any]], **kwargs) -> None:
        # Botones interactivos simples
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient_id,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": text[:1024]},
                "action": {"buttons": buttons[:3]}
            }
        }
        await self._post(payload)


class WhatsAppCloudInput(InputChannel):
    """Recibe webhooks de WhatsApp Cloud y entrega a Rasa"""

    def __init__(self, verify_token, access_token, phone_number_id):
        self.verify_token = verify_token
        self.access_token = access_token
        self.phone_number_id = phone_number_id

    @classmethod
    def name(cls):
        return "whatsapp_cloud"

    @classmethod
    def from_credentials(cls, credentials):
        return cls(
            credentials.get("verify_token"),
            credentials.get("access_token"),
            credentials.get("phone_number_id"),
        )

    def blueprint(self, on_new_message):
        bp = Blueprint("whatsapp_webhook", __name__)

        @bp.get("/webhook")
        async def health(request: Request):
            # Verificación del webhook (GET)
            mode = request.args.get("hub.mode")
            token = request.args.get("hub.verify_token")
            challenge = request.args.get("hub.challenge")
            if mode == "subscribe" and token == self.verify_token:
                return response.text(challenge, status=200)
            return response.text("Forbidden", status=403)

        @bp.post("/webhook")
        async def webhook(request: Request):
            try:
                data = request.json or {}
                entry = (data.get("entry") or [{}])[0]
                change = (entry.get("changes") or [{}])[0]
                value = change.get("value") or {}

                # mensajes entrantes
                messages = value.get("messages") or []
                for m in messages:
                    sender = m.get("from")            # E.164 sin "+"
                    mtype = m.get("type")
                    text = ""
                    if mtype == "text":
                        text = (m.get("text") or {}).get("body", "")
                    elif mtype == "interactive":
                        # botones/listas
                        interactive = m.get("interactive") or {}
                        if interactive.get("type") == "button_reply":
                            text = (interactive.get("button_reply") or {}).get("title", "")
                        elif interactive.get("type") == "list_reply":
                            text = (interactive.get("list_reply") or {}).get("title", "")
                    else:
                        # otros tipos (imagen, audio, etc.)
                        text = f"[{mtype}]"

                    # Crea el canal de salida para responder por WhatsApp
                    out = WhatsAppCloudOutput(self.access_token, self.phone_number_id)

                    # Entrega el mensaje a Rasa
                    user_msg = UserMessage(
                        text=text,
                        output_channel=out,
                        sender_id=sender,        
                        input_channel=self.name()
                    )
                    await on_new_message(user_msg)

            except Exception as e:
                logger.exception(f"Error processing webhook: {e}")

            return response.text("OK", status=200)

        return bp


# Helper para cargar desde credentials.yml o variables de entorno
def from_credentials(env: Dict[Text, Any]) -> WhatsAppCloudInput:
    verify_token = env.get("verify_token") or os.getenv("VERIFY_TOKEN")
    access_token = env.get("access_token") or os.getenv("WHATSAPP_TOKEN")
    phone_number_id = env.get("phone_number_id") or os.getenv("PHONE_NUMBER_ID")

    if not (verify_token and access_token and phone_number_id):
        raise ValueError("Faltan verify_token / access_token / phone_number_id para WhatsApp Cloud")

    return WhatsAppCloudInput(
        verify_token=verify_token,
        access_token=access_token,
        phone_number_id=phone_number_id
    )
