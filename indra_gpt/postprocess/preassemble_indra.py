import logging
from typing import List
from indra.statements import Statement, RegulateActivity, activity_types
from indra.pipeline.pipeline import AssemblyPipeline
from indra.tools import assemble_corpus as ac
from indra.preassembler.grounding_mapper.gilda import ground_statements
import indra.statements as ist

logger = logging.getLogger(__name__)

class INDRAPreassembler:
    def __init__(self, grounding: bool = True, grounding_strategy: str = "gilda"):
        """
        Prepares INDRA statements by applying grounding, filtering, and preassembly.

        Args:
            grounding (bool): Whether to apply grounding (default: True)
            grounding_strategy (str): Strategy to use for grounding (default: 'gilda')
        """
        self.grounding = grounding
        self.grounding_strategy = grounding_strategy

    def preassemble(self, stmts: List[Statement]) -> List[Statement]:
        """
        Apply the full preassembly pipeline on input INDRA statements.

        Args:
            stmts (List[Statement]): List of INDRA Statement objects.

        Returns:
            List[Statement]: Final list of preassembled statements.
        """
        if not stmts:
            logger.warning("No statements provided for preassembly.")
            return []

        pipeline = AssemblyPipeline()
        pipeline.append(ac.filter_no_hypothesis)
        pipeline.append(ac.filter_no_negated)
        pipeline.append(self.custom_ground_statements, self.grounding, self.grounding_strategy)
        pipeline.append(ac.filter_grounded_only)
        pipeline.append(ac.filter_genes_only)
        pipeline.append(ac.filter_human_only)
        pipeline.append(self.filter_curation_supported_types)
        pipeline.append(self.filter_RegulateActivity_only_valid_activities)
        pipeline.append(ac.run_preassembly, return_toplevel=False)

        logger.info(f"Running preassembly pipeline on {len(stmts)} statements...")
        final_stmts = pipeline.run(stmts)
        logger.info(f"Finished preassembly: {len(final_stmts)} statements retained.")
        return final_stmts

    @staticmethod
    def custom_ground_statements(stmts: List[Statement],
                                 grounding: bool,
                                 strategy: str) -> List[Statement]:
        if not grounding:
            logger.info("Skipping grounding step.")
            return stmts

        if strategy.lower() != "gilda":
            raise ValueError(f"Unsupported grounding strategy: {strategy}")

        logger.info(f"Applying GILDA grounding to {len(stmts)} statements...")
        grounded = ground_statements(stmts)
        logger.info(f"Grounded {len(grounded)} statements.")
        return grounded

    @staticmethod
    def filter_curation_supported_types(stmts: List[Statement]) -> List[Statement]:
        """
        Filters out statements not supported for curation or review.
        """
        supported = []
        for stmt in stmts:
            cls = type(stmt)
            if ((issubclass(cls, ist.Modification) and cls is not ist.Modification) or
                (issubclass(cls, ist.RegulateAmount) and cls is not ist.RegulateAmount) or
                (issubclass(cls, ist.RegulateActivity) and cls is not ist.RegulateActivity) or
                issubclass(cls, (ist.Autophosphorylation, ist.Association, ist.Complex,
                                 ist.ActiveForm, ist.Translocation,
                                 ist.Gef, ist.Gap, ist.Conversion))):
                supported.append(stmt)
        return supported

    @staticmethod
    def filter_RegulateActivity_only_valid_activities(stmts: List[Statement]) -> List[Statement]:
        """
        For RegulateActivity statements, keep only those with valid activity types.
        """
        filtered = []
        for stmt in stmts:
            if isinstance(stmt, RegulateActivity):
                if stmt.obj_activity in activity_types:
                    filtered.append(stmt)
            else:
                filtered.append(stmt)
        return filtered
