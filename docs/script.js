var setCurrentStep = function(new_step, length) {
    var be = new_step < this.current_step ? 'instant' : 'smooth';
    this.current_step = new_step;
    var p = this.$refs.steps[this.current_step].parentNode;
    var s = p.scrollLeftMax / (length - 1);
    p.scrollTo({ left: s * new_step, behavior: be });
};

Vue.component('steps-carousel', {
    props: ['length', 'name', 'estimated_rewards'],
    template: '#steps-carousel-template',
    data() {
      return {
        current_step: 0,
      }
    },
    methods: {
      setStep: setCurrentStep
    }
});

var v = new Vue({
    el: '#content',
    data: {
      estimated_rewards: {
        'simple-example-1': [
          [0.999],
          [0.996, 0.001],
          [0.996, 0.0],
          [0.994, 0.003],
          [0.991, 0.003],
          [0.987, 0.001],
          [0.981, 0.019],
          [0.973, 0.006],
          [0.985, 0.028],
          [1.0],
        ],
        'complex-manipulation-1': [
          [0.975],
          [0.972, 0.084],
          [0.774],
          [0.839],
          [0.858, 0.015],
          [0.788, 0.076],
          [1.0],
        ],
        'push-try-1': [
          [0.899],
          [0.983, 0.047],
          [0.969, 0.006],
          [0.989, 0.042],
        ],
        'push-try-3': [
          [0.912],
          [0.981, 0.003],
          [0.879],
          [0.976, 0.007],
          [0.985, 0.04],
        ],
        'bin-picking-2': [
          [0.991],
          [0.995, 0.001],
          [0.981, 0.012],
          [0.983, 0.005],
          [0.971, 0.014],
          [0.930, 0.037],
          [0.949, 0.025],
        ]
      }
    }
});