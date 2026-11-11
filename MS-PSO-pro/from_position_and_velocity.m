% Function to create particle from position and velocity
function particle = from_position_and_velocity(position, velocity)
    particle.position = position;
    particle.velocity = velocity;
    particle.personal_best_position = position;
    particle.personal_best_fitness = [];
end