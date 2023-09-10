from typing import List
from enum import Enum
import os

class ONVMWorkerType(Enum):
    """Enum to distinguish the different NF types"""
    DividerNF = 1
    ForwarderNF = 2
    Unknown = 0

class RunConfig(object):
    def __init__(self) -> None:
        super().__init__()


class DPDKConfig(object):
    """Represents the DPDK configuration for ONVM instances."""

    def __init__(self, 
                 corelist: str,
                 memory_channels: int,
                 portmask: int
    ) -> None:
        """Initialises the DPDK configuration for ONVM instances.

        Args:
            corelist: List of cores available for DPDK.
            memory_channels: Maximum number of memory channels for DPDK.
            portmask: Hesadecimal mask of the NIC ports to use.

        Returns:
            None
        """
        super().__init__()
        self.corelist = corelist
        self.memory_channels = memory_channels
        self.portmask = portmask

    @property
    def corelist(self) -> str:
        return self._corelist

    @corelist.setter
    def corelist(self, value: str):
        assert type(value) is str, 'corelist value is not a string'
        self._corelist = value

    @property
    def memory_channels(self) -> int:
        return self._memory_channels

    @memory_channels.setter
    def memory_channels(self, value: int):
        assert type(value) is int, 'memory_channels value is not an int'
        self._memory_channels = value

    @property
    def portmask(self) -> int:
        return self._portmask

    @portmask.setter
    def portmask(self, value: int):
        assert type(value) is int, 'portmask value is not an int'
        self._portmask = value

    def to_dict(self) -> dict:
        """Return the DPDK config as dict"""
        return {
            "corelist": self._corelist,
            "memory_channels": self._memory_channels,
            "portmask": self._portmask 
        }


class ONVMConfig(object):
    """Represents the ONVM configuration"""

    def __init__(self, 
                 output: str,
                 service_id: int,
                 instance_id: int
    ) -> None:
        """Initialises the ONVM configuration.

        Args:
            output: The output channel to use: web or stdout.
            service_id: The service id of the object.
            instance_id: The instance id of the object.

        Returns:
            None
        """
        super().__init__()
        self.output = output
        self.service_id = service_id
        self.instance_id = instance_id

    @property
    def output(self) -> str:
        return self._output

    @output.setter
    def output(self, value: str):
        assert type(value) is str, 'output value is not a string'
        self._output = value

    @property
    def service_id(self) -> int:
        return self._service_id

    @service_id.setter
    def service_id(self, value: int):
        assert type(value) is int, 'service_id value is not an int'
        self._service_id = value

    @property
    def instance_id(self) -> int:
        return self._instance_id

    @instance_id.setter
    def instance_id(self, value: int):
        assert type(value) is int, 'instance_id value is not an int'
        self._instance_id = value

    def to_dict(self) -> dict:
        """Return the ONVM config as dict"""
        return {
            "output": self._output,
            "serviceid": self._service_id,
            "instanceid": self._instance_id
        }


class ONVMManagerConfig(object):
    """Configuration for the ONVM Manager"""
    def __init__(self,
                 corelist: str,
                 memory_channels: int,
                 portmask: int,
                 coremask: str,
                 logging_dir: str,
                 output: str,
                 num_of_tx: int = 1,
                 tx_assigment: List[List[int]] = None,
                 tx_mode: int = 0
    ) -> None:
        """Configuration for the ONVM Manager

        Args:
            corelist: List of cores available for DPDK.
            memory_channels: Maximum number of memory channels for DPDK.
            portmask: Hexadecimal mask of the NIC ports to use for DPDK.
            coremask: Mask for ONVM of which cores can be used for NFs (e.g. '0xF8')
            logging_dir: Folder for stats files
            output: The output channel to use: web or stdout.
            num_of_tx: Number of how many TX-Threads should be used.
            tx_assignment: List with List of instance id for each TX-Thread.
                The first List corresponds with the first TX-Thread, the second with the second...
            tx_mode: [0,1] 0: NF Manager handles Stats. 1: TX-Thread handles Stats
        """
        super().__init__()
        self.dpdk_config = DPDKConfig(corelist, memory_channels, portmask)
        self.coremask = coremask
        self.output = output
        self.logging_dir = logging_dir
        self.num_of_tx = num_of_tx
        self.tx_assignment = tx_assigment
        self.portmask = portmask
        self.tx_mode = tx_mode

    @property
    def tx_mode(self) -> int:
        return self._tx_mode

    @tx_mode.setter
    def tx_mode(self, value: int):
        #assert type(value) is int, 'tx_mode is not an int.'
        self._tx_mode = value

    @property
    def portmask(self) -> int:
        return self._portmask

    @portmask.setter
    def portmask(self, value: int):
        self._portmask = value

    @property
    def dpdk_config(self) -> DPDKConfig:
        return self._dpdk_config

    @dpdk_config.setter
    def dpdk_config(self, value: DPDKConfig):
        assert type(value) is DPDKConfig, 'dpdk_config value is not a DPDKConfig'
        self._dpdk_config = value

    @property
    def coremask(self) -> str:
        return self._coremask

    @coremask.setter
    def coremask(self, value: str):
        assert type(value) is str, 'coremask value is not a string'
        self._coremask = value

    @property
    def output(self) -> str:
        return self._output

    @output.setter
    def output(self, value: str):
        assert type(value) is str, 'output value is not a string'
        self._output = value

    @property
    def logging_dir(self) -> str:
        return self._logging_dir

    @logging_dir.setter
    def logging_dir(self, value: str):
        assert type(value) is str, 'logging_dir value is not a string'
        self._logging_dir = value

    @property
    def num_of_tx(self) -> int:
        return self._num_of_tx

    @num_of_tx.setter
    def num_of_tx(self, value: int):
        assert type(value) is int, 'num_of_tx value is not an int'
        self._num_of_tx = value

    @property
    def tx_assignment(self) -> List[List[int]]:
        return self._tx_assignment

    @tx_assignment.setter
    def tx_assignment(self, value: List[List[int]]):
        if value is None:
            self._tx_assignment = None
        else: 
            assert len(value) > 0, "tx_assignment value was empty"
            self._tx_assignment = value

    def to_dict(self) -> dict:
        """Returns the ONVMManager config as dict"""
        config_dict = {}
        config_dict["dpdk"] = self.dpdk_config.to_dict()
        config_dict["onvm"] = {
            "output": self.output,
            "coremask": self.coremask,
            "portmask": self.portmask,
            "logging_directory": self.logging_dir,
            "num_tx": self.num_of_tx,
            "tx_mode": self.tx_mode
        }

        if self.tx_assignment:
            tx_dict = {}
            for i, val in enumerate(self.tx_assignment):
                tx_dict[str(i)] = val
            
            config_dict["tx_assignment"] = tx_dict

        return config_dict


class ONVMWorkerConfig(object):
    """Baseclass for NF config classes"""

    def __init__(self, type: ONVMWorkerType,
                 output: str,
                 service_id: int,
                 instance_id: int
    ) -> None:
        """ Initialises the base config with the given values
        
        Args:
            output: The output stream to use.
            service_id: ServiceID for ONVM.
            instance_id: InstanceID for ONVM.
        """
        super().__init__()
        self._type = type
        # Required by DPDK, but has no real use
        self.dpdk_config = DPDKConfig('0,1', 4, 0)
        self.onvm_config = ONVMConfig(output, service_id, instance_id)

    @property
    def dpdk_config(self) -> DPDKConfig:
        return self._dpdk_config

    @dpdk_config.setter
    def dpdk_config(self, value: DPDKConfig):
        assert type(value) is DPDKConfig, 'dpdk_config value is not a DPDKConfig'
        self._dpdk_config = value

    @property
    def onvm_config(self) -> ONVMConfig:
        return self._onvm_config

    @onvm_config.setter
    def onvm_config(self, value: ONVMConfig):
        assert type(value) is ONVMConfig, 'onvm_config value is not an ONVMConfig'
        self._onvm_config = value

    def get_type(self) -> ONVMWorkerType:
        """Returns the type of the NF"""
        return self._type
    
    def to_dict(self) -> None:
        raise "Implement in child class"


class DividerNFConfig(ONVMWorkerConfig):
    def __init__(self,
                 instance_id: int,
                 service_id: int,
                 output: str = 'stdout',
                 core: int = 0
    ) -> None:
        """Represents the config of a DividerNF.
        
        Args:
            instance_id: InstanceID for ONVM.
            service_id: ServiceID for ONVM.
            output: Output stream for ONVM.
            core: The core on which the NF should run on.
                Core = 0 equals standard ONVM core allocation
        """
        super().__init__(ONVMWorkerType.DividerNF, output, service_id, instance_id)
        self.core = core

    @property
    def core(self) -> int:
        return self._core

    @core.setter
    def core(self, value: int):
        assert type(value) is int, 'core value is not an int'
        self._core = value

    def to_dict(self) -> dict:
        """Returns the config of the DividerNF as dict"""
        config_dict = {}
        config_dict["dpdk"] = self.dpdk_config.to_dict()
        config_dict["onvm"] = self.onvm_config.to_dict()
        config_dict["nf"] = {
            "run_on_core": self._core
        }
        return config_dict


class ForwarderNFConfig(ONVMWorkerConfig):
    def __init__(self,
                 instance_id: int,
                 service_id: int,
                 destination: int,
                 output: str = 'stdout',
                 core: int = 0,
                 iteration_mean: int = 25,
                 iteration_var: float = 0.0,
                 array_size: int = 0,
                 use_nic: int = 0,

    ) -> None:
        """Represents the config of a DividerNF.
        
        Args:
            instance_id: InstanceID for ONVM.
            service_id: ServiceID for ONVM.
            destination: ServiceID of the next NF in an SFC.
            output: Output stream for ONVM.
            core: The core on which the NF should run on.
                Core = 0 equals standard ONVM core allocation
            iteration_mean: Mean of normal distribution for dummy loops
            iteration_var: Variance of normal distribution for dummy loops
            array_size: Size of array for cache accesses testing. Size represents
                the number of uint64_t elements in the array
            use_nic: if a number is provided for use_nic, the NF will send packets
                out trough the NIC port with the given id
        """
        super().__init__(ONVMWorkerType.ForwarderNF, output, service_id, instance_id)
        self.core = core
        self.destination = destination
        self.iteration_mean = iteration_mean
        self.iteration_var = iteration_var
        self.array_size = array_size
        self.use_nic = use_nic

    @property
    def core(self) -> int:
        return self._core

    @core.setter
    def core(self, value: int):
        assert type(value) is int, 'core value is not an int'
        self._core = value

    @property
    def destination(self) -> int:
        return self._destination

    @destination.setter
    def destination(self, value: int):
        assert type(value) is int, 'destination value is not an int'
        self._destination = value

    @property
    def iteration_mean(self) -> int:
        return self._iteration_mean

    @iteration_mean.setter
    def iteration_mean(self, value: int):
        assert type(value) is int, 'iteration_mean value is not an int'
        self._iteration_mean = value

    @property
    def iteration_var(self) -> float:
        return self._iteration_var

    @iteration_var.setter
    def iteration_var(self, value: float):
        assert type(value) is float, 'iteration_var value is not a float'
        self._iteration_var = value

    @property
    def array_size(self) -> int:
        return self._array_size

    @array_size.setter
    def array_size(self, value: int):
        assert type(value) is int, 'array_size value is not an int'
        self._array_size = value

    @property
    def use_nic(self) -> int:
        return self._use_nic

    @use_nic.setter
    def use_nic(self, value: int):
        assert type(value) is int, 'use_nic value is not an int'
        self._use_nic = value

    def to_dict(self) -> dict:
        """Returns the config of a ForwarderNF as dict"""
        config_dict = {}
        config_dict["dpdk"] = self.dpdk_config.to_dict()
        config_dict["onvm"] = self.onvm_config.to_dict()
        config_dict["nf"] = {
            "destination": self.destination,
            "run_on_core": self.core,
            "iteration_mean": self.iteration_mean,
            "iteration_var": self.iteration_var,
            "array_size": self.array_size
        }

        if self._use_nic:
            config_dict['nf']['use_nic'] = self.use_nic
        
        return config_dict


class AutomatorConfig(object):
    def __init__(self, 
                manager: ONVMManagerConfig, 
                worker_list: List[ONVMWorkerConfig],
                weights: List[float] = [],
                first_ids: List[int] = [],
                rate: float = 2e6,
                packet_size: int = 64,
                num_tx: int = 4
        ) -> None:
        """Represents the Configuration for the Automator.

        Args:
            manager: ONVMManagerConfig for the manager.
            worker_list: List of ONVMWorkerConfigs containing the 
                config for all NFs.
        """
        super().__init__()
        self.manager = manager
        self.worker_list = worker_list
        self.weights = weights
        self.first_ids = first_ids
        self.rate = rate
        self.packet_size = packet_size
        self.delay_in_ms = 0
        self.test_duration = 10
        self.result_dir = ''
        self.num_tx = num_tx

        if os.environ['HOME']:
            self.logging_dir = os.environ['HOME']
            self.result_dir = os.environ['HOME']
        else:
            self.logging_dir = ''

    @property
    def manager(self) -> ONVMManagerConfig:
        return self._manager

    @manager.setter
    def manager(self, value: ONVMManagerConfig):
        assert type(value) is ONVMManagerConfig, 'manager is value is not ONVMManagerConfig'
        self._manager = value

    @property
    def worker_list(self) -> List[ONVMWorkerConfig]:
        return self._worker_list

    @worker_list.setter
    def worker_list(self, value: List[ONVMWorkerConfig]):
        self._worker_list = value

    @property
    def delay_in_ms(self) -> int:
        return self._delay_in_ms

    @delay_in_ms.setter
    def delay_in_ms(self, value: int):
        assert type(value) is int, 'delay_in_ms value is not int'
        self._delay_in_ms = value
    
    @property
    def test_duration(self) -> int:
        return self._test_duration

    @test_duration.setter
    def test_duration(self, value: int):
        assert type(value) is int, 'test_duration value is not int'
        self._test_duration = value

    @property
    def logging_dir(self) -> str:
        return self._logging_dir

    @logging_dir.setter
    def logging_dir(self, value: str):
        assert type(value) is str, 'logging_dir value is not str'
        self._logging_dir = value

    @property
    def result_dir(self) -> str:
        return self._result_dir

    @result_dir.setter
    def result_dir(self, value: str):
        assert type(value) is str, 'result_dir value is not str'
        self._result_dir = value

    @property
    def weights(self) -> List[float]:
        return self._weights

    @weights.setter
    def weights(self, value: List[float]):
        self._weights = value

    @property
    def first_ids(self) -> List[int]:
        return self._first_ids

    @first_ids.setter
    def first_ids(self, value: List[int]):
        self._first_ids = value

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, value: float):
        self._rate = value
    
    @property
    def packet_size(self) -> int:
        return self._packet_size

    @packet_size.setter
    def packet_size(self, value: int):
        self._packet_size = value

    @property
    def num_tx(self) -> int:
        return self._num_tx

    @num_tx.setter
    def num_tx(self, value: int):
        assert type(value) is int, 'num_tx value is not int'
        self._num_tx = value

    def to_dict(self):
        result = {}
        result["General"] = {
            "delay_in_ms": self.delay_in_ms,
            "duration": self.test_duration,
            "logging_dir": self.logging_dir,
            "result_dir": self.result_dir,
            "weights": self.weights,
            "first_ids": self.first_ids,
            "rate": self.rate,
            "packet_size": self.packet_size,
            "num_tx": self.num_tx
        }
        result["Manager"] = self.manager.to_dict()
        result["NFs"] = [worker.to_dict() for worker in self.worker_list]

        return result