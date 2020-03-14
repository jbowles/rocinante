extern crate dirs;
extern crate failure;

use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use rust_bert::SentimentClassifier;

use std::io::ErrorKind;
use std::path::PathBuf;
// use std::sync::Arc;
use tch::Device;

#[actix_rt::main]
async fn main() -> std::io::Result<()> {
    /*
    let home_dir: PathBuf = dirs::home_dir().unwrap();
    let on_device = Device::cuda_if_available();
    let model = match setup_sentiment_model(home_dir, on_device) {
        Err(e) => return Err(e),
        Ok(m) => m,
    };
    let data = web::Data::new(Arc::new(model));
    */

    HttpServer::new(|| {
        App::new()
            // .app_data(data.clone())
            .route("/", web::get().to(index))
            .route("/what", web::get().to(what_is_rocinante))
            .route("/sentiment/{input}", web::get().to(get_sentiment))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

async fn index() -> impl Responder {
    HttpResponse::Ok().body("Welcome to Rocinante!")
}
async fn what_is_rocinante() -> impl Responder {
    HttpResponse::Ok().body("Rocinante is Don Quixote's horse.")
}
/*
async fn polarity(
    req: HttpRequest,
    model: web::Data<Mutex<SentimentClassifier>>,
) -> impl Responder {
    let mut model = model.lock().unwrap();
    let input = req.match_info().get("input").unwrap_or("bad input");
    let res = model.predict((&[input]).to_vec());
    HttpResponse::Ok().body(format!("{:?}", res)).await
}
*/
async fn get_sentiment(req: HttpRequest) -> impl Responder {
    let home_dir: PathBuf = dirs::home_dir().unwrap();
    let on_device = Device::cuda_if_available();
    let model = match setup_sentiment_model(home_dir, on_device) {
        Err(e) => return Err(e),
        Ok(m) => m,
    };
    let input = req.match_info().get("input").unwrap_or("bad input");
    let res = model.predict((&[input]).to_vec());
    Ok(HttpResponse::Ok().body(format!("{:?}", res)).await)
}

/*
struct SentO {
    model: SentimentClassifier,
    home_dir: PathBuf,
    on_device: Device,
}

impl Responder for SentO {
    // let home_dir: PathBuf = dirs::home_dir().unwrap();
    // let on_device = Device::cuda_if_available();
    // let sentiment_classifier = setup_sentiment_model(home_dir, on_device)?;
    fn respond_to(self, _req: &HttpRequest) -> Self::Future {
        let input = _req.match_info().get("input").unwrap_or("bad input");
        let res = self.model.predict((&[input]).to_vec());

    HttpResponse::Ok().body(format!("{:?}", res))
    }
}
*/

fn setup_sentiment_model(
    mut model_path: PathBuf,
    d: tch::Device,
) -> Result<SentimentClassifier, std::io::Error> {
    model_path.push("rustbert");
    model_path.push("distilbert_sst2");
    let vocab = &model_path.as_path().join("vocab.txt");
    let config = &model_path.as_path().join("config.json");
    let weights = &model_path.as_path().join("model.ot");
    // Ok(SentimentClassifier::new(vocab, config, weights, d)?)
    match SentimentClassifier::new(vocab, config, weights, d) {
        Err(e) => Err(std::io::Error::new(ErrorKind::Other, e)),
        Ok(m) => Ok(m),
    }
}
