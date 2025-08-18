from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# Importa las funciones de tu script RAG
from actions.rag_actions import connect_milvus_lite, Collection, SentenceTransformer, SEARCH_PARAMS, OLLAMA_MODEL, call_ollama_chat, build_system_prompt, build_user_prompt
from pymilvus import utility, connections

class ActionChatWithRAG(Action):

    def name(self) -> Text:
        return "action_chat_with_rag"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        question = tracker.latest_message.get("text")
        top_k = 5  # Puedes ajustar este valor
        temperature = 0.1
        num_ctx = 2048

        # Conecta a Milvus Lite si no está conectado
        if not connections.has_connection("default"):
            connect_milvus_lite()
        
        if not utility.has_collection("bago_pdf"):
            dispatcher.utter_message(text="No existe la colección. Ingresa primero un PDF.")
            return []

        col = Collection("bago_pdf")
        col.load()

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        qvec = embedder.encode([question]).tolist()

        results = col.search(
            qvec,
            anns_field="embedding",
            param=SEARCH_PARAMS,
            output_fields=["text"],
            limit=top_k
        )

        first = results[0]

        try:
            hits_list = list(first)
        except TypeError:
            hits_list = first

        contexts = []
        distances = []
        for h in hits_list:
            txt = ""
            try:
                ent = getattr(h, "entity", None)
                if ent is not None and hasattr(ent, "get"):
                    txt = ent.get("text") or ""
            except Exception:
                pass
            contexts.append(txt)
            distances.append(float(getattr(h, "distance", 0.0)))

        system = build_system_prompt()
        user = build_user_prompt(question, contexts)

        try:
            answer = call_ollama_chat(
                model=OLLAMA_MODEL,
                system=system,
                user=user,
                options={"temperature": temperature, "num_ctx": num_ctx}
            )

            dispatcher.utter_message(text=answer)

        except Exception as e:
            dispatcher.utter_message(text=f"Fallo al llamar a Ollama. Detalle: {str(e)}")

        return []

class ActionPreguntaDosis(Action):
    def name(self) -> Text:
        return "action_pregunta_dosis"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        medication = next(tracker.get_latest_entity_values("medicamento"), None)
        print(f"Entidad medicamento: {medication}")  # Agrega esta línea
        if not medication:
            dispatcher.utter_message(text="Por favor, especifica el medicamento.")
            return []

        # Conecta a Milvus Lite si no está conectado
        if not connections.has_connection("default"):
            connect_milvus_lite()

        # Ensure collection exists
        if not utility.has_collection(COLLECTION_NAME):
            dispatcher.utter_message(text=f"No existe la colección {COLLECTION_NAME}. Ingresa primero un PDF.")
            return []

        col = Collection(COLLECTION_NAME)
        col.load()

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        # Modifica la pregunta para buscar información de dosis
        question = f"dosis de {medication}"
        qvec = embedder.encode([question]).tolist()

        results = col.search(
            qvec,
            anns_field="embedding",
            param=SEARCH_PARAMS,
            output_fields=["text"],
            limit=3  # Ajusta el límite según sea necesario
        )

        # Extrae los contextos relevantes
        contexts = [hit.entity.get("text", "") for hit in results[0]]

        if contexts:
            dosage_info = "\n".join(contexts)  # Combina los contextos
            dispatcher.utter_message(text=f"La dosis de {medication} es: {dosage_info}")
        else:
            dispatcher.utter_message(text=f"No se encontró información de dosis para {medication}.")

        return []