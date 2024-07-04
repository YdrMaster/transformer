#![doc = include_str!("../README.md")]

mod manager;
mod response;
mod schemas;

use causal_lm::CausalLM;
use http_body_util::{combinators::BoxBody, BodyExt, Empty};
use hyper::{
    body::{Bytes, Incoming},
    server::conn::http1,
    service::Service as HyperService,
    Method, Request, Response, StatusCode,
};
use hyper_util::rt::TokioIo;
use manager::ServiceManager;
use response::{error, success, text_stream};
use std::{
    future::Future,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    pin::Pin,
    sync::Arc,
};
use tokio::net::TcpListener;
use tokio_stream::wrappers::UnboundedReceiverStream;

#[macro_use]
extern crate log;

pub async fn start_infer_service<M>(
    service: service::Service<M>,
    port: u16,
    session_capacity: Option<usize>,
) -> std::io::Result<()>
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send,
{
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    info!("start service at {addr}");

    let app = App(Arc::new(ServiceManager::new(service, session_capacity)));
    let listener = TcpListener::bind(addr).await?;
    loop {
        let app = app.clone();
        let (stream, _) = listener.accept().await?;
        tokio::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(TokioIo::new(stream), app)
                .await
            {
                warn!("Error serving connection: {err:?}");
            }
        });
    }
}

struct App<M: CausalLM>(Arc<ServiceManager<M>>);

impl<M: CausalLM> Clone for App<M> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<M> HyperService<Request<Incoming>> for App<M>
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send,
{
    type Response = Response<BoxBody<Bytes, hyper::Error>>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: Request<Incoming>) -> Self::Future {
        let manager = self.0.clone();

        macro_rules! response {
            ($method:ident; $f:expr) => {
                Box::pin(async move {
                    let whole_body = req.collect().await?.to_bytes();
                    let req = serde_json::from_slice(&whole_body);
                    Ok(match req {
                        Ok(req) => match manager.$method(req) {
                            Ok(ret) => $f(ret),
                            Err(e) => error(e),
                        },
                        Err(e) => error(schemas::Error::WrongJson(e)),
                    })
                })
            };
        }

        match (req.method(), req.uri().path()) {
            (&Method::POST, "/infer") => {
                response!(infer; |ret| text_stream(UnboundedReceiverStream::new(ret)))
            }
            (&Method::POST, "/fork") => response!(fork ; success),
            (&Method::POST, "/drop") => response!(drop_; success),
            // Return 404 Not Found for other routes.
            _ => Box::pin(async move {
                Ok(Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(
                        Empty::<Bytes>::new()
                            .map_err(|never| match never {})
                            .boxed(),
                    )
                    .unwrap())
            }),
        }
    }
}
