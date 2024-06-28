//! All HttpResponses in this App.

use middlewares::utils;
use http_body_util::{combinators::BoxBody, BodyExt, Full, StreamBody};
use hyper::{
    body::{Bytes, Frame},
    header::CONTENT_TYPE,
    Response, StatusCode,
};
use serde::Serialize;
use tokio_stream::{Stream, StreamExt};

pub fn text_stream(
    s: impl Stream<Item = String> + Send + Sync + 'static,
) -> Response<BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .body(StreamBody::new(s.map(|s| Ok(Frame::data(s.into())))).boxed())
        .unwrap()
}

pub fn success(success: impl utils::Success) -> Response<BoxBody<Bytes, hyper::Error>> {
    #[derive(Serialize)]
    struct SuccessResponse<'a> {
        message: &'a str,
    }
    let body = SuccessResponse {
        message: success.msg(),
    };

    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "application/json")
        .body(full(serde_json::to_string(&body).unwrap()))
        .unwrap()
}

pub fn error(e: utils::Error) -> Response<BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(e.status())
        .header(CONTENT_TYPE, "application/json")
        .body(full(serde_json::to_string(&e.body()).unwrap()))
        .unwrap()
}

#[inline]
fn full(chunk: impl Into<Bytes>) -> BoxBody<Bytes, hyper::Error> {
    Full::new(chunk.into())
        .map_err(|never| match never {})
        .boxed()
}
