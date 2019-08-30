Vue.component('steps-carousel', {
    props: ['name', 'steps'],
    template: '#steps-carousel-template',
    data() {
      return {
        current_step: 0,
      }
    },
    methods: {
      setStep: function(new_step, length) {
        var be = new_step < this.current_step ? 'instant' : 'smooth';
        this.current_step = new_step;
        var p = this.$refs.steps[this.current_step].parentNode;
        var s = p.scrollLeftMax / (length - 1);
        $(p).animate({ scrollLeft: s * new_step }, 300);
      }
    }
});

var v = new Vue({
    el: '#content',
    data: {
      simple_example_1: [
        { action_type: 1, reward: 1, estimated_reward: 0.999 },
        { action_type: 1, reward: 1, estimated_reward: 0.996, estimated_reward_std: 0.001 },
        { action_type: 0, reward: 1, estimated_reward: 0.996, estimated_reward_std: 0.0 },
        { action_type: 1, reward: 1, estimated_reward: 0.994, estimated_reward_std: 0.003 },
        { action_type: 1, reward: 1, estimated_reward: 0.991, estimated_reward_std: 0.003 },
        { action_type: 0, reward: 1, estimated_reward: 0.987, estimated_reward_std: 0.001 },
        { action_type: 2, reward: 1, estimated_reward: 0.981, estimated_reward_std: 0.019 },
        { action_type: 2, reward: 1, estimated_reward: 0.973, estimated_reward_std: 0.006 },
        { action_type: 1, reward: 1, estimated_reward: 0.985, estimated_reward_std: 0.028 },
        { action_type: 4, reward: 1, estimated_reward: 1.0 },
      ],
      complex_manipulation_1: [
        { action_type: 0, reward: 1, estimated_reward: 0.975 },
        { action_type: 1, reward: 1, estimated_reward: 0.972, estimated_reward_std: 0.084 },
        { action_type: 3, reward: 0.62, estimated_reward: 0.774 },
        { action_type: 3, reward: 0.77, estimated_reward: 0.839 },
        { action_type: 1, reward: 1, estimated_reward: 0.858, estimated_reward_std: 0.015 },
        { action_type: 1, reward: 1, estimated_reward: 0.788, estimated_reward_std: 0.076 },
        { action_type: 4, reward: 1, estimated_reward: 1.0 },
      ],
      bin_picking_2: [
        { action_type: 1, reward: 1, estimated_reward: 0.991 },
        { action_type: 0, reward: 1, estimated_reward: 0.995, estimated_reward_std: 0.001 },
        { action_type: 1, reward: 1, estimated_reward: 0.981, estimated_reward_std: 0.012 },
        { action_type: 1, reward: 1, estimated_reward: 0.983, estimated_reward_std: 0.005 },
        { action_type: 0, reward: 1, estimated_reward: 0.971, estimated_reward_std: 0.014 },
        { action_type: 0, reward: 1, estimated_reward: 0.930, estimated_reward_std: 0.037 },
        { action_type: 0, reward: 1, estimated_reward: 0.949, estimated_reward_std: 0.025 },
      ],
      push_try_1: [
        { action_type: 3, reward: 0.91, estimated_reward: 0.899 },
        { action_type: 1, reward: 1, estimated_reward: 0.983, estimated_reward_std: 0.047 },
        { action_type: 0, reward: 1, estimated_reward: 0.969, estimated_reward_std: 0.006 },
        { action_type: 1, reward: 1, estimated_reward: 0.989, estimated_reward_std: 0.042 },
      ],
      push_try_3: [
        { action_type: 3, reward: 0.84, estimated_reward: 0.912 },
        { action_type: 1, reward: 1, estimated_reward: 0.981, estimated_reward_std: 0.003 },
        { action_type: 3, reward: 0.89, estimated_reward: 0.879 },
        { action_type: 1, reward: 1, estimated_reward: 0.976, estimated_reward_std: 0.007 },
        { action_type: 1, reward: 1, estimated_reward: 0.985, estimated_reward_std: 0.04 },
      ],
    }
});