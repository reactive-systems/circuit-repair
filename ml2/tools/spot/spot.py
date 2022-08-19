"""Spot"""

import logging

from ...globals import IMAGE_BASE_NAME
from ...ltl.ltl_sat import LTLSatStatus
from ...trace import TraceMCStatus
from ..containerized_grpc_service import ContainerizedGRPCService
from ..protos import ltl_pb2
from . import spot_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_NAME = IMAGE_BASE_NAME + "/spot-grpc-server:latest"


class Spot(ContainerizedGRPCService):
    def __init__(self, cpu_count: int = 1, mem_limit: str = "2g", port: int = None):
        super().__init__(IMAGE_NAME, cpu_count, mem_limit, port, service_name="Spot")
        self.stub = spot_pb2_grpc.SpotStub(self.channel)

    def check_sat(self, formula: str, simplify: bool = False, timeout: int = None):
        pb_problem = ltl_pb2.LTLSatProblem(formula=formula, simplify=simplify, timeout=timeout)
        pb_solution = self.stub.CheckSat(pb_problem)
        return LTLSatStatus(pb_solution.status), pb_solution.trace

    def mc_trace(self, formula: str, trace: str, timeout: int = None):
        pb_problem = ltl_pb2.LTLTraceMCProblem(formula=formula, trace=trace, timeout=timeout)
        pb_solution = self.stub.MCTrace(pb_problem)
        return TraceMCStatus(pb_solution.status)
