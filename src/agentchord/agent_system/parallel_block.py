from copy import deepcopy
from typing import List, Optional, Union

from ..environment import BaseEnvironment
from ..metadata import BaseMetaData
from .base_agent_system import BaseAgentSystem


class ParallelBlock(BaseAgentSystem):
    """
    A block that contains multiple agent systems that can run in parallel.
    """

    def __init__(self,
        system_name: str,
        environment: BaseEnvironment,
        subsystems: List[BaseAgentSystem],
        on_start_actions: Optional[List[dict]] = None,
        on_completion_actions: Optional[List[dict]] = None,
        tools: Optional[List[str]] = None,
        maximum_loops: int = 50,
        log_name: str = "",
        return_list: bool = False
    ) -> None:
        super().__init__(
            system_name=system_name,
            environment=environment,
            tools=tools,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        self.return_list = return_list
        for subsystem in subsystems:
            self.add_subsystem(subsystem)
        if on_start_actions:
            for action in on_start_actions:
                self.add_on_start_action(
                    action["subsystem_name"],
                    action["action_name"],
                    action["action_function"]
                )
        if on_completion_actions:
            for action in on_completion_actions:
                self.add_on_completion_action(
                    action["subsystem_name"],
                    action["action_name"],
                    action["action_function"]
                )

    def execution_loop(
            self,
            meta_data: BaseMetaData,
            loop: bool = False,
        ) -> Union[BaseMetaData, List[BaseMetaData]]:
        loop_counter = 0
        meta_data_list = []
        final_output = []
        final_note = []
        final_tool = []
        while loop_counter < self.maximum_loops:
            self.subsystem_sequence.set_not_done()
            while not self.subsystem_sequence.is_done():
                # Move subsystem sequence forward
                current_subsystem_name = self.subsystem_sequence.get_current_subsystem_name()
                self.subsystem_sequence.update_next_subsystem()
                # Make a copy of the meta data for each subsystem
                new_meta_data = deepcopy(meta_data)
                # Trigger on-start events
                if current_subsystem_name in self.on_start_actions:
                    new_meta_data = self.on_start_actions[current_subsystem_name](new_meta_data)
                # Trigger child completion event
                new_meta_data = self.subsystems[current_subsystem_name].execution_loop(new_meta_data)
                # Trigger on-completion event
                if current_subsystem_name in self.on_completion_actions:
                    new_meta_data = self.on_completion_actions[current_subsystem_name](new_meta_data)
                # Update the final meta data
                meta_data_list.append(new_meta_data)
                final_output.append(new_meta_data.output)
                final_note.append(new_meta_data.note)
                final_tool.extend(new_meta_data.tool)
                # Early termination if the environment is done
                if self.environment.is_done():
                    self.subsystem_sequence.set_done()
            # Only the root system controls the looping
            if self.parent_number > 0 or not loop or self.environment.is_done():
                break
            loop_counter += 1
        if self.return_list:
            return meta_data_list
        else:
            new_meta_data.output = final_output
            new_meta_data.note = final_note
            new_meta_data.tool = final_tool
            return new_meta_data