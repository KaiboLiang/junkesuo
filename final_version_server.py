import os, sys
from typing import Any

from fastapi import FastAPI, Response, status, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# from flyagent.logs import logger

import argparse

from fastapi.responses import HTMLResponse, JSONResponse
import traceback
import final_version
import json


def create_app():
    # 全局异常处理
    class CustomExceptionMiddleware:
        def __init__(self, app: FastAPI) -> None:
            self.app = app

        async def __call__(self, request: Request, call_next) -> Any:
            try:
                return await call_next(request)
            except Exception as exc:
                return await self.handle_exception(exc)

        @classmethod
        async def handle_exception(cls, exc: Exception) -> JSONResponse:
            # logger.error(traceback.format_exc())
            print(traceback.format_exc())
            # 获取异常信息
            exc_type, exc_value, exc_traceback = sys.exc_info()
            # 提取错误类型和值
            error_info = f"{exc_type.__name__}: {exc_value}"
            return JSONResponse(
                status_code=500,
                content={
                    "response": f"{error_info}",
                }
            )

    # def create_sample_network_data(safety_cost_params, transfer_point_params, other_params):
    #     根据企业节点坐标、需求点坐标、中转点坐标，计算这些个坐标之间距离
    #
    # def create_sample_transport_params(network_data, transport_params, time_params, other_params):
    #     根据企业到需求点的距离，选择运输方式和中转点

    @app.middleware("http")
    async def custom_exception_handler(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            return await CustomExceptionMiddleware.handle_exception(exc)

    @app.post("/test")
    async def test(request: Request):
        request_json = await request.json()

        return JSONResponse(content={"response": test}, status_code=200)


    @app.post("/input_time_parameters")
    async def input_time_parameters(request: Request):
        request_json = await request.json()
        request_json = json.dumps(request_json, ensure_ascii=False)
        response_json = final_version.input_time_parameters(request_json)
        return JSONResponse(content={"response": response_json}, status_code=200)

    @app.post("/input_transport_parameters")
    async def input_transport_parameters(request: Request):
        request_json = await request.json()
        request_json = json.dumps(request_json, ensure_ascii=False)
        response_json = final_version.input_transport_parameters(request_json)
        return JSONResponse(content={"response": response_json}, status_code=200)

    @app.post("/input_safety_cost_parameters")
    async def input_safety_cost_parameters(request: Request):
        request_json = await request.json()
        request_json = json.dumps(request_json, ensure_ascii=False)
        response_json = final_version.input_safety_cost_parameters(request_json)
        return JSONResponse(content={"response": response_json}, status_code=200)

    @app.post("/input_transfer_point_parameters")
    async def input_transfer_point_parameters(request: Request):
        request_json = await request.json()
        request_json = json.dumps(request_json, ensure_ascii=False)
        response_json = final_version.input_transfer_point_parameters(request_json)
        return JSONResponse(content={"response": response_json}, status_code=200)

    @app.post("/input_other_parameters")
    async def input_other_parameters(request: Request):
        request_json = await request.json()

        # 检查是否是嵌套格式（向后兼容）
        if "other_input" in request_json:
            # 如果有嵌套结构，按原来的方式处理
            other_input = json.dumps(request_json["other_input"], ensure_ascii=False)
            safety_cost_params = request_json.get("safety_cost_params")
            transfer_point_params = request_json.get("transfer_point_params")
        else:
            # 如果没有嵌套结构，直接使用整个请求作为输入
            other_input = json.dumps(request_json, ensure_ascii=False)
            safety_cost_params = None
            transfer_point_params = None

        response_json = final_version.input_other_parameters(other_input, safety_cost_params, transfer_point_params)
        return JSONResponse(content={"response": response_json}, status_code=200)

    @app.post("/input_weight_to_plan")
    async def input_weight_to_plan(request: Request):
        request_json = await request.json()
        request_json = json.dumps(request_json, ensure_ascii=False)
        response_json = final_version.input_weight_to_plan(request_json)
        return JSONResponse(content={"response": response_json}, status_code=200)

    return app


# 单位转换, 定量指标计算, 定性， 匹配对应字段， 数据库目前没有定量和定性的区分

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a server on a specific port.')
    parser.add_argument('--port', type=int, required=False, help='The port number to run the server on.')
    args = parser.parse_args()
    api_port = args.port

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app = create_app()
    api_host = os.environ.get("API_HOST", "0.0.0.0")
    if api_port is None:
        # api_port = int(os.environ.get("API_PORT", config.main.main_server_port))
        api_port = 16389
    print("Visit http://localhost:{}/docs for API document.".format(api_port))
    uvicorn.run(app, host=api_host, port=api_port)

    # cnmon
