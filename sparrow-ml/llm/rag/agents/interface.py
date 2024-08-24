from abc import ABC, abstractmethod
from typing import Any
from typing import List
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Abstract Interface
class Pipeline(ABC):
    @abstractmethod
    def run_pipeline(
        self,
        payload: str,
        query_inputs: list[str] | None = None,
        query_types: list[str] | None = None,
        keywords: list[str] | None = None,
        query: str | None = None,
        file_path: str | None = None,
        index_name: str | None = None,
        options: List[str] | None = None,
        group_by_rows: bool = True,
        update_targets: bool = True,
        debug: bool = False,
        local: bool = True,
    ) -> Any:
        pass


# Factory Method
def get_pipeline(agent_name: str) -> Pipeline:
    if agent_name == "llamaindex":
        from rag.agents.llamaindex.llamaindex import LlamaIndexPipeline

        return LlamaIndexPipeline()
    elif agent_name == "haystack":
        from rag.agents.haystack.haystack import HaystackPipeline

        return HaystackPipeline()
    elif agent_name == "vllamaindex":
        from rag.agents.llamaindex.vllamaindex import VLlamaIndexPipeline

        return VLlamaIndexPipeline()
    elif agent_name == "vprocessor":
        from rag.agents.llamaindex.vprocessor import VProcessorPipeline

        return VProcessorPipeline()
    elif agent_name == "fcall":
        from rag.agents.instructor.fcall import FCall

        return FCall()
    elif agent_name == "instructor":
        from rag.agents.instructor.instructor import InstructorPipeline

        return InstructorPipeline()
    elif agent_name == "unstructured-light":
        from rag.agents.unstructured.unstructured_light import UnstructuredLightPipeline

        return UnstructuredLightPipeline()
    elif agent_name == "unstructured":
        from rag.agents.unstructured.unstructured import UnstructuredPipeline

        return UnstructuredPipeline()
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
