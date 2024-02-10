# vta-rag

LlamaIndex based RAG code.

## Notes

Okay, their `npx create-llama` is meant to framework a _complete_ RAG-driven chat app. See <https://www.npmjs.com/package/create-llama>. And if that wasn't enough, they have a complete one: <https://github.com/run-llama/chat-llamaindex>.

At this point, I am doubting if our idea is worth doing, since its much faster to work off or modify one of the above. Unless we do RAG in some novel way from scratch, but that would mean examining all current frameworks for lessons learnt to design something better. That's too much work.

## Doc Links

- [`ServiceContext`](https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context.html): Config singleton for the "engine" of LlamaIndex from node splitting strategy, to retrieval & generation.
- [`StorageContext`](https://docs.llamaindex.ai/en/stable/api_reference/storage.html): Config singleton for the "storage" of LlamaIndex, like where to dump documents, nodes, embeddings, index metadata, etc.

## Dump of Promising Links

- Knowledge Graph Index: <https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphDemo.html>
- Composing multiple indexes: <https://docs.llamaindex.ai/en/stable/examples/composable_indices/ComposableIndices.html>
