from collections.abc import Callable

from openhands.sdk import Event
from openhands.sdk.conversation import ConversationState

from crg_ce.graph.data_types import CENode, ConfidenceGraph

"""
BaseNodeGenerator defines the signature which all node generators will conform to
"""
BaseNodeGenerator = Callable[[ConversationState, list[Event]], list[CENode]]

"""
BaseGraphGenerator defines the signature which all graph generators will conform to
"""
BaseGraphGenerator = Callable[[ConversationState, list[Event]], ConfidenceGraph]
