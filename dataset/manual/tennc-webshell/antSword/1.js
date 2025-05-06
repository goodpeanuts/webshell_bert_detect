module.exports = (pwd, data) => {
  data[pwd] = new Buffer(data['_']).toString('base64');
  delete data['_'];
  return data;
}
