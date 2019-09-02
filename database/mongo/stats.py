from collections import defaultdict

from data.loader import Loader

database_list = Loader.get_databases()
total_count = sum(Loader.get_episode_count(d) for d in database_list)
print(f'Total count {total_count}')

recordings_on_date = defaultdict(lambda: 0)
image_count = 0

for d in database_list:
    for _, e in Loader.yield_episodes(d):
        data = e['id'].split('-')
        date = f'{data[0]}-{data[1]}' # -{data[2]}'
        recordings_on_date[date] += 1
        image_count += len(e['actions'][0]['images'])

print(f'Total image count {image_count}')
for m in sorted(recordings_on_date):
    print(f'{m}: {recordings_on_date[m]}')
