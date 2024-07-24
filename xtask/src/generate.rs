use crate::{print_now, InferenceArgs, Task};
use causal_lm::CausalLM;
use service::Service;
use std::{fmt::Debug, path::Path, time::Instant};

#[derive(Args, Default)]
pub(crate) struct GenerateArgs {
    #[clap(flatten)]
    pub inference: InferenceArgs,
    /// Prompt.
    #[clap(long, short)]
    pub prompt: String,
    /// Max number of steps to generate.
    #[clap(long)]
    pub max_steps: Option<usize>,
}

impl Task for GenerateArgs {
    #[inline]
    fn inference(&self) -> &InferenceArgs {
        &self.inference
    }

    async fn typed<M>(self, meta: M::Meta)
    where
        M: CausalLM + Send + Sync + 'static,
        M::Storage: Send,
        M::Error: Debug,
    {
        let (service, _handle) = Service::<M>::load(&self.inference.model, meta);

        let prompt = if Path::new(&self.prompt).is_file() {
            println!("prompt from file: {}", self.prompt);
            std::fs::read_to_string(&self.prompt).unwrap()
        } else {
            self.prompt
        };
        print_now!("{}", prompt);

        let max_steps = self.max_steps.unwrap_or(usize::MAX);
        let mut steps = 0;
        let mut generator = service.generate(&*prompt, Some(self.inference.sample_args()));

        let mut time: Option<Instant> = None;
        while let Some(s) = generator.decode().await {
            time.get_or_insert_with(Instant::now);
            match &*s {
                "\\n" => println!(),
                _ => print_now!("{s}"),
            }
            steps += 1;
            if steps == max_steps {
                break;
            }
        }
        if let Some(time) = time {
            println!();
            println!("Time elapsed: {:?}/tok", time.elapsed().div_f32(steps as f32));
        };

    }
}
