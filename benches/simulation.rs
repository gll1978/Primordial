//! Performance benchmarks for PRIMORDIAL

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use primordial::{Config, World};
use primordial::neural::NeuralNet;

fn benchmark_world_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("world_step");

    for population in [100, 500, 1000].iter() {
        let mut config = Config::default();
        config.organisms.initial_population = *population;
        config.world.grid_size = 80;

        let mut world = World::new_with_seed(config, 42);

        // Warm up
        world.run(10);

        group.bench_with_input(
            BenchmarkId::new("population", population),
            population,
            |b, _| {
                b.iter(|| {
                    world.step();
                });
            },
        );
    }

    group.finish();
}

fn benchmark_neural_forward(c: &mut Criterion) {
    let net = NeuralNet::new_minimal(24, 10);
    let inputs = [0.5f32; 24];

    c.bench_function("neural_forward_minimal", |b| {
        b.iter(|| {
            net.forward(black_box(&inputs))
        });
    });

    let mut complex_net = NeuralNet::new_with_instincts(24, 10);
    for _ in 0..5 {
        complex_net.add_neuron();
    }

    c.bench_function("neural_forward_complex", |b| {
        b.iter(|| {
            complex_net.forward(black_box(&inputs))
        });
    });
}

fn benchmark_sensing(c: &mut Criterion) {
    let mut config = Config::default();
    config.organisms.initial_population = 500;
    let world = World::new_with_seed(config.clone(), 42);

    let org = &world.organisms[0];

    c.bench_function("organism_sense", |b| {
        b.iter(|| {
            org.sense(
                &world.food_grid,
                &world.spatial_index,
                &world.organisms,
                world.time,
                &world.config,
            )
        });
    });
}

fn benchmark_mutation(c: &mut Criterion) {
    use primordial::neural::MutationConfig;

    let mutation_config = MutationConfig::default();

    c.bench_function("neural_mutation", |b| {
        let mut net = NeuralNet::new_minimal(24, 10);
        b.iter(|| {
            net.mutate(&mutation_config);
        });
    });
}

fn benchmark_checkpoint(c: &mut Criterion) {
    let mut config = Config::default();
    config.organisms.initial_population = 1000;
    let mut world = World::new_with_seed(config, 42);
    world.run(100);

    let checkpoint = world.create_checkpoint();

    c.bench_function("checkpoint_serialize", |b| {
        b.iter(|| {
            bincode::serialize(black_box(&checkpoint)).unwrap()
        });
    });

    let serialized = bincode::serialize(&checkpoint).unwrap();

    c.bench_function("checkpoint_deserialize", |b| {
        b.iter(|| {
            let _: primordial::checkpoint::Checkpoint =
                bincode::deserialize(black_box(&serialized)).unwrap();
        });
    });
}

criterion_group!(
    benches,
    benchmark_world_step,
    benchmark_neural_forward,
    benchmark_sensing,
    benchmark_mutation,
    benchmark_checkpoint,
);

criterion_main!(benches);
