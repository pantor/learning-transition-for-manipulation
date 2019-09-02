var app = new Vue({
  el: '#app',
  delimiters: ["[[", "]]"],
  data: {
    database: '',
    database_list: [],
    current_id: '',
    page_start_index: 0,
    page_index: 0,
    row_count: 4,
    column_count: 4,
    page_count: 0,
    episodes: [],
    page_episodes: [],
    stats: {},
    detail: {},
    last_action: {},
    draw_bin: 1,
    suffix: 'ed-v',
    image_suffixes: ['ed-v', 'rd-v', 'rc-v', 'ed-after', 'rd-after', 'rc-after', 'rc-file'],
    show_second_image: false,
    filter: {
      reward: -1,
      final_d_lower: 0,
      final_d_upper: 0.1,
      id: '',
    },
    show_settings: false,
  },
  filters: {
    round: function(value, decimals) {
      if (!value) { value = 0; }
      if (!decimals) { decimals = 0; }
      return Math.round(value * Math.pow(10, decimals)) / Math.pow(10, decimals);
    }
  },
  methods: {
    loadDatabase: function(event) {
      localStorage.setItem('database', app.database);

      $.get("/api/episodes", {database: app.database, ...app.filter}, (data) => {
        app.episodes = data.reverse();

        app.page_count = Math.ceil(app.episodes.length / (app.row_count * app.column_count));
        app.updatePage();

        if (app.episodes.length > 0) {
          app.current_id = app.episodes[0].id;
          app.updateDetail();

          $.get("/api/episode/" + app.current_id, {"database": app.database}, (data) => {
            app.last_action = data;
          });
        }
      });

      $.get("/api/stats", {database: app.database, suffix: app.suffix}, (data) => {
        app.stats = data;
      });
    },
    updateDetail: function(event) {
      $.get("/api/episode/" + app.current_id, {"database": app.database}, (data) => {
        app.detail = data;

        const images = app.lastAction(app.detail).images;
        //if (!images.includes(app.suffix)) {
        //  app.suffix = images[0];
        //}
      });
    },
    updatePage: function(event) {
      localStorage.setItem('column_count', app.column_count);
      localStorage.setItem('draw_bin', app.draw_bin);

      if (app.episodes) {
        app.page_episodes = app.episodes.slice(app.page_start_index, app.page_start_index + app.row_count * app.column_count);
      }

      $.get("/api/stats", {database: app.database, suffix: app.suffix}, (data) => {
        app.stats = data;
      });

      setTimeout(app.resize, 250);
      setTimeout(app.resize, 1000);
    },
    diffIndex: function(index_diff) {
      index_diff = parseInt(index_diff);
      let index = app.episodes.findIndex((e) => { return e.id == app.current_id; });
      if (0 <= index + index_diff && index + index_diff < app.episodes.length) {
        index += index_diff;
        app.current_id = app.episodes[index].id;

        if (index > app.page_start_index + app.row_count * app.column_count - 1) {
          app.nextPage();
        } else if (index < app.page_start_index) {
          app.prevPage();
        }
        app.updateDetail();
      }
    },
    setDetailId: function(id) {
      app.current_id = id;
      app.updateDetail();
    },
    nextPage: function(event) {
      if (app.page_index < app.page_count - 1) {
        app.page_index += 1;
        app.page_start_index += app.row_count * app.column_count;
        app.updatePage();
      }
    },
    prevPage: function(event) {
      if (app.page_index > 0) {
        app.page_index -= 1;
        app.page_start_index -= app.row_count * app.column_count;
        app.updatePage();
      }
    },
    setPage: function(index) {
      app.page_index = index;
      app.page_start_index = app.page_index * app.row_count * app.column_count;
      app.updatePage();
    },
    lastAction: function(episode) {
      return episode.actions.slice(-1)[0];
    },
    deleteEpisode: function() {
      $.post("/api/delete/" + app.current_id, { database: app.database }, (data) => {
        let index = app.episodes.findIndex((e) => { return e.id == app.current_id; });
        app.episodes.splice(index, 1);
        app.current_id = app.episodes[Math.min(index, app.episodes.length - 1)].id;
        app.page_start_index = app.page_index * app.row_count * app.column_count;
        app.updatePage();
        app.updateDetail();
      });
    },
    resize: function(event) {
      const viewpoint_height = document.documentElement.clientHeight - 50;
      const container_height = document.getElementById('image-container').clientHeight;
      const image_element = document.getElementById('0');

      let image_height = 26;
      if (image_element) {
        image_height = Math.max(image_element.clientHeight, image_height);
      }

      const new_row_count = Math.floor(viewpoint_height / image_height);

      if (app.row_count != new_row_count) {
        app.row_count = new_row_count;
        localStorage.setItem('row_count', app.row_count);

        app.page_count = Math.ceil(app.episodes.length / (app.row_count * app.column_count));
        app.updatePage();
      }
    },
  }
})

var socket = io();

window.addEventListener('resize', app.resize)

app.row_count = localStorage.getItem('row_count') || app.row_count;
app.column_count = localStorage.getItem('column_count') || app.column_count;
app.draw_bin = localStorage.getItem('draw_bin') || app.draw_bin;

$.get("/api/database-list",(data) => {
  app.database_list = data;
  app.database = localStorage.getItem('database') || data[0];
  app.loadDatabase();
});

socket.on('new-episode', function (data) {
  app.last_action = data;

  if (data.database == app.database) {
    app.episodes.unshift(data);
    app.page_count = Math.ceil(app.episodes.length / (app.row_count * app.column_count));
    app.updatePage();
    app.updateDetail();

    $.get("/api/stats", {database: app.database}, (data) => {
      app.stats = data;
    });
  }
});

socket.on('new-attempt', function (data) {
  app.last_action = data;
});

$("body").keydown(function(e) {
  if(e.keyCode == 37) { // left
    app.diffIndex(-1);
  } else if(e.keyCode == 38) { // up
    app.diffIndex(-app.column_count);
  } else if(e.keyCode == 39) { // right
    app.diffIndex(1);
  } else if(e.keyCode == 40) { // down
    app.diffIndex(app.column_count);
  }
});

$('body').bind('wheel', function (event) {
  if (event.originalEvent.deltaY < 0) {
    app.diffIndex(-1);
  } else {
    app.diffIndex(1);
  }
});
