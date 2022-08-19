"""gRPC Server that checks the satisfiability of an LTL formula and verifies traces using Spot"""

import argparse
from concurrent import futures
import logging
import time

import grpc

from . import spot_pb2_grpc
from .spot_wrapper import automaton_trace, mc_trace
from ..protos import ltl_pb2

logger = logging.getLogger("Spot gRPC Server")


class SpotServicer(spot_pb2_grpc.SpotServicer):
    def CheckSat(self, request, context):
        start = time.time()
        solution = automaton_trace(request.formula, request.simplify, request.timeout)
        end = time.time()
        print(f"Checking Satisfiability took {end - start} seconds")
        return ltl_pb2.LTLSatSolution(
            status=solution["status"].value, trace=solution.get("trace", None)
        )

    def MCTrace(self, request, context):
        start = time.time()
        solution = mc_trace(request.formula, request.trace, request.timeout)
        end = time.time()
        logger.info("Model checking trace took %d seconds" % end - start)
        return ltl_pb2.TraceMCSolution(status=solution.value)


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    spot_pb2_grpc.add_SpotServicer_to_server(SpotServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spot gRPC server")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=50051,
        metavar="port number",
        help=("port on which server accepts RPCs"),
    )
    args = parser.parse_args()
    serve(args.port)
