"""Start the installed MATLAB Engine, evaluate a scalar expression, and exit."""
import json
import matlab.engine


engine = matlab.engine.start_matlab()
try:
    result = engine.eval("1+2", nargout=1)
    version = engine.version(nargout=1)
    print(json.dumps({"engine_start": True, "eval_1_plus_2": result, "matlab_version": version}))
finally:
    engine.quit()
