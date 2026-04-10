
db = db.getSiblingDB('admin');

db = db.getSiblingDB('mle_db');
db.createCollection('predictions');

db.grantRolesToUser('root', ['readWrite']);

db.predictions.createIndex({ timestamp: -1 });

print('MongoDB initialization completed!');
print('Database: mle_db');
print('Collection: predictions');
print('User root granted readWrite on mle_db');

