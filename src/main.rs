extern crate dirs;
extern crate failure;
use rust_bert::SentimentClassifier;
use std::io::prelude::*;
use std::io::ErrorKind;
use std::net::TcpListener;
use std::net::TcpStream;
use std::path::PathBuf;
use tch::Device;

fn main() -> std::io::Result<()> {
    let home_dir: PathBuf = dirs::home_dir().unwrap();
    let on_device = Device::cuda_if_available();
    let model = match setup_sentiment_model(home_dir, on_device) {
        Err(e) => return Err(e),
        Ok(m) => m,
    };
    let input = [
        "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
        "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
        "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
    ];

    let listener = TcpListener::bind("127.0.0.1:8080").unwrap();
    for stream in listener.incoming() {
        let stream = stream.unwrap();
        handle_connection(stream, (&input).to_vec(), &model)
    }

    Ok(())
}

fn handle_connection(mut stream: TcpStream, input: Vec<&str>, model: &SentimentClassifier) {
    let output = model.predict(input.to_vec());
    for sentiment in output {
        println!("{:?}", sentiment);
    }
    let mut buffer = [0; 512];
    stream.read_exact(&mut buffer).unwrap();
    println!("Request: {}", String::from_utf8_lossy(&buffer[..]));
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
