var app = new Vue({
  el: '#app',
  delimiters: ["[[", "]]"],
  data: {
    database: '',
    database_list: [],
    last_episode: {},
    last_action: {},
    draw_bin: 1,
    suffix: 'ed-v'
  },
  filters: {
    round: function(value, decimals) {
      if (!value) { value = 0; }
      if (!decimals) { decimals = 0; }
      return Math.round(value * Math.pow(10, decimals)) / Math.pow(10, decimals);
    }
  },
  methods: {
    loadLastAction: function(event) {
      localStorage.setItem('database', app.database);

      $.get("/api/episodes", {database: app.database}, (data) => {
        app.last_episode = data.slice(-1)[0];

        $.get("/api/episode/" + app.last_episode.id, {"database": app.database}, (data) => {
          app.last_action = app.lastAction(data);
          app.last_action.database = app.database;
        });
      });
    },
    lastAction: function(episode) {
      return episode.actions.slice(-1)[0];
    }
  }
})

var socket = io();

app.draw_bin = localStorage.getItem('draw_bin') || app.draw_bin;

$.get("/api/database-list",(data) => {
  app.database_list = data;
  app.database = localStorage.getItem('database') || data[0];
  app.loadLastAction();
});

socket.on('new-episode', function (data) {
  app.last_episode.id = data.id;
  app.last_action = app.lastAction(data);
});

socket.on('new-attempt', function (data) {
  console.log(data);
  app.last_episode.id = data.episode_id;
  app.database = data.database;
  app.last_action = data.action;
});
