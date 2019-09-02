#include <frankr/gripper.hpp>


Gripper::Gripper(const std::string& fci_ip): franka::Gripper(fci_ip), fci_ip(fci_ip) { }

Gripper::Gripper(const std::string& fci_ip, double gripper_speed): franka::Gripper(fci_ip), fci_ip(fci_ip), gripper_speed(gripper_speed) { }

double Gripper::width() const {
  auto state = ((franka::Gripper*) this)->readOnce();
  return state.width + width_calibration;
}

bool Gripper::homing() const {
  return ((franka::Gripper*) this)->homing();
}

bool Gripper::stop() const {
  return ((franka::Gripper*) this)->stop();
}

bool Gripper::isGrasping() const {
  const double current_width = this->width();
  const bool libfranka_is_grasped = ((franka::Gripper*) this)->readOnce().is_grasped;
  const bool width_is_grasped = std::abs(current_width - last_clamp_width) < 0.003; // [m], magic number
  const bool width_larger_than_threshold = current_width > 0.005; // [m]
  return libfranka_is_grasped && width_is_grasped && width_larger_than_threshold;
}

bool Gripper::move(double width) { // [m]
  try {
    return ((franka::Gripper*) this)->move(width - width_calibration, gripper_speed); // [m] [m/s]
  } catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    this->stop();
    this->homing();
    return ((franka::Gripper*) this)->move(width - width_calibration, gripper_speed); // [m] [m/s]
  }
}

std::future<bool> Gripper::moveAsync(double width) { // [m]
  return std::async(std::launch::async, &Gripper::move, this, width - width_calibration);
}

bool Gripper::open() {
  return move(max_width);
}

bool Gripper::clamp() {
  const bool success = this->grasp(min_width, gripper_speed, gripper_force, min_width, 1.0); // [m] [m/s] [N] [m] [m]
  last_clamp_width = this->width();
  return success;
}

bool Gripper::release(double width) { // [m]
  try {
    this->stop();
    return this->move(width);
  } catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    this->homing();
    this->stop();
    return this->move(width);
  }
}
