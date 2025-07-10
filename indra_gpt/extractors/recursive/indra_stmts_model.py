from __future__ import annotations
from pydantic import BaseModel, Field

from typing import List, Optional
from pydantic import BaseModel, Field

class ModCondition(BaseModel):
    model_config = {
        "description": "Post-translational modification state at an amino acid position."
    }

    mod_type: Optional[str] = Field(
        None,
        pattern=r"^((phosphorylation)|(ubiquitination)|(sumoylation)|(hydroxylation)|(acetylation))$",
        description="The type of post-translational modification, e.g., 'phosphorylation'."
    )
    residue: Optional[str] = Field(
        None,
        description="The modified amino acid, e.g., 'Y' or 'tyrosine'."
    )
    position: Optional[str] = Field(
        None,
        description="the position of the modified amino acid, e.g., '202'."
    )
    is_modified: Optional[bool] = Field(
        None,
        description="Whether the modification is present or absent."
    )


class MutCondition(BaseModel):
    model_config = {
        "description": "Mutation state of an amino acid position of an Agent."
    }

    position: Optional[str] = Field(
        None, 
        description="Residue position of the mutation in the protein sequence."
    )
    residue_from: Optional[str] = Field(
        None, 
        description="Wild-type (unmodified) amino acid residue at the given position."
    )
    residue_to: Optional[str] = Field(
        None, 
        description="Amino acid at the position resulting from the mutation."
    )


class ActivityCondition(BaseModel):
    model_config = {
        "description": "An active or inactive state of a protein."
    }

    activity_type: Optional[str] = Field(
        None,
        pattern=(
            r"^((activity)|(transcription)|(catalytic)|(gtpbound)|"
            r"(kinase)|(phosphatase)|(gef)|(gap))$"
        ),
        description="The type of activity, e.g. 'kinase'."
    )
    is_active: Optional[bool] = Field(
        None,
        description="Whether the activity is present or absent."
    )


class BoundCondition(BaseModel):
    model_config = {
        "description": "Identify Agents bound (or not bound) to a given Agent in a given context."
    }
    agent: Optional[Agent] = Field(
        None,
        description="A molecular entity, e.g., a protein."
    )
    is_bound: Optional[bool] = Field(
        None,
        description="Whether the given Agent is bound or unbound in the current context."
    )


class DBRef(BaseModel):
    model_config = {
        "description": "Database references for a molecular entity."
    }

    TEXT: Optional[str] = Field(
        None,
        description="Term of the molecular entity, directly taken from the text."
    )


class Agent(BaseModel):
    model_config = {
        "description": "A molecular entity, e.g., a protein."
    }

    name: Optional[str] = Field(
        None,
        description="The name of the agent, preferably a canonicalized name such as an HGNC gene name."
    )
    db_refs: Optional[DBRef] = Field(
        None,
        description="Dictionary of database identifiers associated with this agent."
    )
    mods: Optional[List[ModCondition]] = Field(
        None,
        description="Modification state of the agent."
    )
    bound_conditions: Optional[List[BoundCondition]] = Field(
        None,
        description="Other agents bound to the agent in this context."
    )
    mutations: Optional[List[MutCondition]] = Field(
        None,
        description= "Amino acid mutations of the agent."
    )
    activity: Optional[ActivityCondition] = Field(
        None,
        description="Activity of the agent."
    )
    location: Optional[str] = Field(
        None,
        description="Cellular location of the agent."
    )
    sbo: Optional[str] = Field(
        None,
        description="Role of this agent in the systems biology ontology"
    )


class Epistemics(BaseModel):
    model_config = {
        "description": "A dictionary describing various forms of epistemic certainty associated with a statement."
    }

    hypothesis: Optional[bool] = Field(
        None,
        description="True if the statement is phrased as a hypothesis or open question."
            
    )
    negation: Optional[bool] = Field(
        None,
        description="True if the statement expresses a negation or denial."
    )
    direct: Optional[bool] = Field(
        None,
        description="True if the statement explicitly describes a direct causal relationship."
    )


class Evidence(BaseModel):
    model_config = {
        "description": "Container for evidence supporting a given statement."
    }

    source_api: Optional[str] = Field(
        None,
        description="INDRA API used to capture the statement, e.g., 'trips', 'biopax', 'bel'."
    )
    source_id: Optional[str] = Field(
        None,
        description=("For statements drawn from databases, "
                     "ID of the database entity corresponding to the statement.")
    )
    pmid: Optional[str] = Field(
        None,
        description="The PMID of the source where the statement was captured."
    )
    text: Optional[str] = Field(
        None,
        description="Natural language text supporting the statement."
    )
    annotations: Optional[dict] = Field(
        None,
        description=("Additional information on the context of the statement, "
                     "e.g., species, cell line, tissue type, etc. The entries "
                     "may vary depending on the source of the information."
        )
    )
    epistemics: Optional[Epistemics] = Field(
        None,
        description=("A dictionary describing various forms of epistemic "
                     "certainty associated with the statement.")
    )


class Statement(BaseModel):
    model_config = {
        "description": "The parent class of all statements."
    }

    evidence: Optional[List[Evidence]] = Field(
        None,
        description="A list of evidence supporting the statement."
    )
    supports: Optional[List[str]] = Field(
        None,
        description="A list of statements that this statement supports."
    )
    supported_by: Optional[List[str]] = Field(
        None,
        description="a list of statements that support this statement."
    )
    uuid: Optional[str] = Field(
        None,
        description="Statement UUID."
    )


class RegulateActivity(Statement):
    model_config = {
        "description": "Regulation of activity (such as activation and inhibition)."
    }

    type: Optional[str] = Field(
        None,
        pattern=r"^((Activation)|(Inhibition))$",
        description='The specific type of the regulation of activity.'
    )
    subj: Optional[Agent] = Field(
        None,
        description="The agent responsible for the change in activity, i.e., the 'upstream' node."
    )
    obj: Optional[Agent] = Field(
        None,
        description="The agent whose activity is influenced by the subject, i.e., the 'downstream' node."
    )
    obj_activity: Optional[str] = Field(
        None,
        pattern=(
            r"^((activity)|(transcription)|(catalytic)|(gtpbound)|"
            r"(kinase)|(phosphatase)|(gef)|(gap))$"
        ),
        description="The activity of the object agent that is affected, e.g., its 'kinase' activity."
    )


class Modification(Statement):
    model_config = {
        "description": "Statement representing the modification of a protein."
    }

    type: Optional[str] = Field(
        None,
        pattern=(
            r"^((Phosphorylation)|(Dephosphorylation)|(Ubiquitination)|"
            r"(Deubiquitination)|(Sumoylation)|(Desumoylation)|(Hydroxylation)|"
            r"(Dehydroxylation)|(Acetylation)|(Deacetylation)|(Glycosylation)|"
            r"(Deglycosylation)|(Farnesylation)|(Defarnesylation)|"
            r"(Geranylgeranylation)|(Degeranylgeranylation)|(Palmitoylation)|"
            r"(Depalmitoylation)|(Myristoylation)|(Demyristoylation)|(Ribosylation)|"
            r"(Deribosylation)|(Methylation)|(Demethylation))$"
        ),
        description='The specific type of the modification.'
    )
    enz: Optional[Agent] = Field(
        None,
        description="The enzyme involved in the modification."
    )
    sub: Optional[Agent] = Field(
        None,
        description="The substrate that is modified."
    )
    residue: Optional[str] = Field(
        None,
        description="The amino acid residue being modified, or None if it is unknown or unspecified."
    )
    position: Optional[str] = Field(
        None,
        description="The position of the modified amino acid, or None if it is unknown or unspecified."
    )


class SelfModification(Statement):
    model_config = {
        "description": "Statement representing the self-modification of a protein."
    }

    type: Optional[str] = Field (
        None,
        pattern=r"^((Autophosphorylation)|(Transphosphorylation))$",
        description="The specific type of the self-modification."
    )
    enz: Optional[Agent] = Field(
        None,
        description="The enzyme involved in the modification."
    )
    residue: Optional[str] = Field(
        None,
        description="The amino acid residue being modified, or None if it is unknown or unspecified."
    )
    position: Optional[str] = Field(
        None,
        description="The position of the modified amino acid, or None if it is unknown or unspecified."
    )


class ActiveForm(Statement):
    model_config = {
        "description": (
            "Specifies conditions causing an Agent to be active or inactive. "
            "Types of conditions influencing a specific type of biochemical activity can include modifications, "
            "bound Agents, and mutations."
        )
    }

    type: Optional[str] = Field(
        None,
        pattern=r"^ActiveForm$",
        description="The specific type of the active form."
    )
    agent: Optional[Agent] = Field(
        None,
        description=(
            "The Agent in a particular active or inactive state. "
            "The sets of ModConditions, BoundConditions, and MutConditions on the given Agent instance "
            "indicate the relevant conditions."
        )
    )
    activity: Optional[str] = Field(
        None,
        description="The type of activity influenced by the given set of conditions, e.g., 'kinase'."
    )
    is_active: Optional[bool] = Field(
        None,
        description="Whether the conditions are activating or not."
    )   


class Gef(Statement):
    model_config = {
        "description": (
            "Exchange of GTP for GDP on a small GTPase protein mediated by a GEF. "
            "Represents the generic process by which a guanosine exchange factor (GEF) "
            "catalyzes nucleotide exchange on a GTPase protein."
        )
    }

    type: Optional[str] = Field(
        None,
        pattern=r"^Gef$",
        description="The specific type of the Gef statement."
    )
    gef: Optional[Agent] = Field(
        None,
        description="The guanosine exchange factor."
    )
    ras: Optional[Agent] = Field(
        None,
        description="The GTPase protein."
    )


class Gap(Statement):
    model_config = {
        "description": (
            "Acceleration of a GTPase protein's GTP hydrolysis rate by a GAP. "
            "Represents the generic process by which a GTPase activating protein (GAP) "
            "catalyzes GTP hydrolysis by a particular small GTPase protein."
        )
    }

    type: Optional[str] = Field(
        None,
        pattern=r"^Gap$",
        description="The specific type of the Gap statement."
    )
    gap: Optional[Agent] = Field(
        None,
        description="The GTPase activating protein."
    )
    ras: Optional[Agent] = Field(
        None,
        description="The GTPase protein."
    )


class Complex(Statement):
    model_config = {
        "description": "A set of proteins observed to be in a bio-molecular complex."
    }

    type: Optional[str] = Field(
        None,
        pattern=r"^((Complex)|(Association))$",
        description="The specific type of the complex statement."
    )
    members: Optional[List[Agent]] = Field(
        None,
        description="A list of members of the bio-molecular complex."
    )

    
class Translocation(Statement):
    model_config = {
        "description": "The translocation of a molecular agent from one location to another."
    }

    type: Optional[str] = Field(
        None,
        pattern=r"^Translocation$",
        description="The specific type of the translocation statement."
    )
    agent: Optional[Agent] = Field(
        None,
        description="The agent which translocates."
    )
    from_location: Optional[str] = Field(
        None,
        description=(
            "The location from which the agent translocates. "
            "This must be a valid GO cellular component name (e.g. 'cytoplasm') or ID (e.g. 'GO:0005737')."
        )
    )
    to_location: Optional[str] = Field(
        None,
        description=(
            "The location to which the agent translocates. "
            "This must be a valid GO cellular component name or ID."
        )
    )


class RegulateAmount(Statement):
    model_config = {
        "description": "Statement representing the regulation of the amount of a molecular agent by another agent."
    }

    type: Optional[str] = Field(
        None,
        pattern=r"^((IncreaseAmount)|(DecreaseAmount))$",
        description="The specific type of the regulation of amount."
    )
    sub: Optional[Agent] = Field(
        None,
        description="The mediating protein."
    )
    obj: Optional[Agent] = Field(
        None,
        description="The affected protein."
    )


class Conversion(Statement):
    model_config = {
        "description": "Conversion of molecular species mediated by a controller protein."
    }

    type: Optional[str] = Field(
        None,
        pattern=r"^Conversion$",
        description="The specific type of the conversion statement."
    )
    subj: Optional[Agent] = Field(
        None,
        description="The protein mediating the conversion."
    )
    obj_from: Optional[List[Agent]] = Field(
        None,
        description="Molecular species being consumed by the conversion."
    )
    obj_to: Optional[List[Agent]] = Field(
        None,
        description="Molecular species being created by the conversion."
    )


class IndraStatements(BaseModel):
    model_config = {
        "description": "All INDRA statements extracted from text."
    }

    regulate_activity_list: Optional[List[RegulateActivity]] = Field(
        None,
        description=(
            "List of RegulateActivity statements mentioned in the input text. "
            "A RegulateActivity statement describes the activation or inhibition of a protein. "
            "This includes event types such as 'Activation' and 'Inhibition', which should be extracted as complete phrases or clauses "
            "from the text (e.g., 'A activates B', 'X inhibits Y')."
        )
    )
    modification_list: Optional[List[Modification]] = Field(
        None,
        description=(
            "List of Modification statements mentioned in the input text. "
            "A Modification statement describes a protein being modified post-translationally, typically by another protein (e.g., a kinase). "
            "Event types in this category include: Phosphorylation, Dephosphorylation, Ubiquitination, Deubiquitination, Sumoylation, Desumoylation, "
            "Hydroxylation, Dehydroxylation, Acetylation, Deacetylation, Glycosylation, Deglycosylation, Farnesylation, Defarnesylation, "
            "Geranylgeranylation, Degeranylgeranylation, Palmitoylation, Depalmitoylation, Myristoylation, Demyristoylation, Ribosylation, "
            "Deribosylation, Methylation, and Demethylation. "
            "Extract complete spans that describe the event, including the modifying enzyme and the target (e.g., 'AKT phosphorylates FOXO3 on Ser253')."
        )
    )
    self_modification_list: Optional[List[SelfModification]] = Field(
        None,
        description=(
            "List of SelfModification statements mentioned in the input text. "
            "A SelfModification statement describes a protein modifying itself, typically through an enzymatic action it catalyzes on its own structure. "
            "Event types in this category include: Autophosphorylation and Transphosphorylation. "
            "Extract full spans that describe the self-modification event, including the protein and site if available "
            "(e.g., 'MAPK1 autophosphorylates on threonine 185 and tyrosine 187')."
        )
    )
    activeform_list: Optional[List[ActiveForm]] = Field(
        None,
        description=(
            "List of ActiveForm statements mentioned in the input text. "
            "An ActiveForm statement describes the specific biochemical conditions under which a protein (agent) is active or inactive. "
            "These conditions may involve modifications, mutations, or binding interactions. "
            "Extract complete spans that indicate an activity state and its associated condition, such as "
            "'AKT is active when phosphorylated at T308' or 'The protein is inactive in its unbound state'."
        )
    )
    gef_list: Optional[List[Gef]] = Field(
        None,
        description=(
            "List of Gef statements mentioned in the input text. "
            "A Gef (Guanine nucleotide Exchange Factor) statement describes a protein facilitating the exchange of GDP for GTP on a small GTPase. "
            "These events typically involve a GEF protein activating a GTPase by promoting nucleotide exchange. "
            "Extract full spans that describe the GEF-mediated exchange event, such as "
            "'SOS promotes GTP loading on RAS' or 'VAV1 facilitates GDP-GTP exchange on RAC1'."
        )
    )
    gap_list: Optional[List[Gap]] = Field(
        None,
        description=(
            "List of Gap statements mentioned in the input text. "
            "A Gap (GTPase-activating protein) statement describes a protein that accelerates the hydrolysis of GTP to GDP on a small GTPase, thereby inactivating it. "
            "These events typically involve a GAP protein increasing the intrinsic GTPase activity of the target. "
            "Extract full spans that describe the GAP-mediated hydrolysis event, such as "
            "'NF1 stimulates GTP hydrolysis on RAS' or 'RASA1 inactivates RAS by accelerating GTP hydrolysis'."
        )
    )
    complex_list: Optional[List[Complex]] = Field(
        None,
        description=(
            "List of Complex statements mentioned in the input text. "
            "A Complex statement describes two or more proteins or molecules physically associating to form a stable complex. "
            "These interactions may be observed experimentally or inferred from co-localization or co-purification data. "
            "Extract full spans that describe the formation or existence of a complex, such as "
            "'BRCA1 forms a complex with BARD1' or 'The MRE11-RAD50-NBS1 complex is essential for DNA repair'."
        )
    )
    translocation_list: Optional[List[Translocation]] = Field(
        None,
        description=(
            "List of Translocation statements mentioned in the input text. "
            "A Translocation statement describes the movement of a molecular agent, such as a protein, from one cellular compartment or location to another. "
            "These events often involve spatial regulation of activity, signaling, or localization. "
            "Extract full spans that describe the translocation event, including origin and destination when available, such as "
            "'FOXO3 translocates from the cytoplasm to the nucleus upon stress' or 'NF-κB moves into the nucleus after stimulation'."
        )
    )
    regulate_amount_list: Optional[List[RegulateAmount]] = Field(
        None,
        description=(
            "List of RegulateAmount statements mentioned in the input text. "
            "A RegulateAmount statement describes one molecular agent influencing the expression level, concentration, or abundance of another. "
            "This includes both upregulation and downregulation events. "
            "Extract full spans that describe these regulatory effects, such as "
            "'IL6 increases expression of STAT3' or 'TP53 suppresses BCL2 levels'."
        )
    )
    conversion_list: Optional[List[Conversion]] = Field(
        None,
        description=(
            "List of Conversion statements mentioned in the input text. "
            "A Conversion statement describes a biochemical transformation in which one or more molecular species are converted into others, typically catalyzed by an enzyme or controller protein. "
            "These events may include metabolic reactions, cleavage, synthesis, or degradation. "
            "Extract full spans that describe such conversions, such as "
            "'Caspase-3 cleaves PARP into two fragments' or 'Enzyme X converts substrate Y into product Z'."
        )
    )

