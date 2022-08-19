"""nuXmv"""

import logging
import os
import random
import subprocess
import shutil
from typing import List, Optional

from ml2.ltl.ltl_spec.ltl_spec import LTLSpec

from ...globals import IMAGE_BASE_NAME
from ...ltl.ltl_mc.ltl_mc_status import LTLMCStatus
from ..containerized_grpc_service import ContainerizedGRPCService
from ..protos import ltl_pb2
from . import nuxmv_pb2, nuxmv_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUXMV_IMAGE_NAME = IMAGE_BASE_NAME + "/nuxmv-grpc-server:latest"


class nuXmv:
    def __init__(self, verify_path: Optional[str] = None, **kargs) -> None:
        if verify_path is None:
            verify_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..",
                    "..",
                    "..",
                    "scripts",
                    "verify.sh",
                )
            )
        if (
            os.path.exists(verify_path)
            and shutil.which("syfco") is not None
            and shutil.which("ltl2smv") is not None
            and shutil.which("smvtoaig") is not None
            and shutil.which("combine-aiger") is not None
            and shutil.which("nuXmv") is not None
        ):
            logger.info("Using local nuXmv installation")
            self.contained_nuXmv: Optional[contained_nuXmv] = contained_nuXmv(
                verify_path=verify_path, **kargs
            )
            self.containerized_nuXmv: Optional[containerized_nuXmv] = None
        else:
            logger.info("Using containerized nuXmv installation")
            self.containerized_nuXmv = containerized_nuXmv(**kargs)
            self.contained_nuXmv = None

    def model_check_batch(
        self,
        specs: List[LTLSpec],
        systems: List[str],
        realizables: List[bool],
        timeout: float = 10.0,
    ) -> List[LTLMCStatus]:
        r = []
        for spec, system, realizable in zip(specs, systems, realizables):
            r.append(self.model_check(spec, system, realizable, timeout))
        return r

    def model_check(self, *args, **kargs) -> LTLMCStatus:
        if self.contained_nuXmv is not None:
            return self.contained_nuXmv.model_check(*args, **kargs)
        elif self.containerized_nuXmv is not None:
            return self.containerized_nuXmv.model_check(*args, **kargs)
        else:
            raise ValueError("model checker not set")


class contained_nuXmv:
    def __init__(
        self,
        temp_dir: str = "/tmp",
        verify_path: str = os.path.join(os.path.expanduser("~"), "verify.sh"),
        **_
    ):
        self.temp_dir = temp_dir
        self.verify_path = verify_path

    def clean(self, temp_dir: str):
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)

    def model_check(
        self,
        spec: LTLSpec,
        system: str,
        realizable: bool = True,
        timeout: float = 10.0,
    ) -> LTLMCStatus:
        identifier = str("%030x" % random.randrange(16 ** 30))
        temp_dir = os.path.join(self.temp_dir, "nuxmv", identifier)
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        spec.to_file(temp_dir, "specification.tlsf", format="tlsf")
        circuit_filepath = os.path.join(temp_dir, "circuit.aag")
        with open(circuit_filepath, "w") as aiger_file:
            aiger_file.write(system)
        try:
            args = [
                self.verify_path,
                circuit_filepath,
                os.path.join(temp_dir, "specification.tlsf"),
            ]
            if realizable:
                args.append("REALIZABLE")
            else:
                args.append("UNREALIZABLE")
            args.append(str(timeout))
            result = subprocess.run(args, capture_output=True, timeout=timeout)
            self.clean(temp_dir)
        except subprocess.TimeoutExpired:
            logging.debug("subprocess timeout")
            self.clean(temp_dir)
            return LTLMCStatus.TIMEOUT
        except subprocess.CalledProcessError:
            logging.error("subprocess called process error")
            self.clean(temp_dir)
            return LTLMCStatus.ERROR
        except Exception as error:
            logging.critical(error)
            self.clean(temp_dir)
        out = result.stdout.decode("utf-8")
        err = result.stderr.decode("utf-8")
        if out == "SUCCESS\n":
            return LTLMCStatus.SATISFIED
        if out == "FAILURE\n":
            return LTLMCStatus.VIOLATED
        if err.startswith("error: cannot read implementation file"):
            return LTLMCStatus.INVALID
        print(out)
        print(err)
        return LTLMCStatus.ERROR


class containerized_nuXmv(ContainerizedGRPCService):
    def __init__(self, cpu_count: int = 2, mem_limit: str = "2g", port: int = None, **_):
        super().__init__(NUXMV_IMAGE_NAME, cpu_count, mem_limit, port, service_name="nuXmv")
        self.stub = nuxmv_pb2_grpc.nuXmvStub(self.channel)

    def model_check(
        self, spec, system: str, realizable: bool = True, timeout: float = 10.0
    ) -> LTLMCStatus:
        specification = ltl_pb2.LTLSpecification(
            inputs=spec.inputs,
            outputs=spec.outputs,
            guarantees=spec.guarantees,
            assumptions=spec.assumptions,
        )
        pb_problem = nuxmv_pb2.Problem(
            specification=specification, system=system, realizable=realizable, timeout=timeout
        )
        pb_solution = self.stub.ModelCheck(pb_problem)
        return LTLMCStatus(nuxmv_pb2.Solution.Status.Name(pb_solution.status).lower())
