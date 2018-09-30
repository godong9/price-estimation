const axios = require('axios');
const moment = require('moment');
const _ = require('lodash');
const fs = require('fs');

const minutes = 60;
const count = 24; // 60분 -> 24개, 10분 -> 144개
const startDate = '2018-01-01';

async function getPriceData(coin, date) {
  const reuslt = await axios.get(`https://crix-api-endpoint.upbit.com/v1/crix/candles/minutes/${minutes}?code=CRIX.UPBIT.KRW-${coin}&count=${count}&to=${date}%2000:00:00`);
  return reuslt.data;
}

async function writeData(coin, dataList) {
  let str = "Date,Open,High,Low,Volume,Close\n";

  const sortedList = _.sortBy(dataList, ['candleDateTime']);
  _.forEach(sortedList, item => {
    str += `${item.candleDateTime},${item.openingPrice},${item.highPrice},${item.lowPrice},${item.candleAccTradeVolume},${item.tradePrice}\n`;
  });

  fs.writeFileSync(`data/${coin}.csv`, str, 'utf8');
}

async function main(coin) {
  const nowDate = moment();
  let dataList = [];

  let targetDate = moment(startDate);

  while (true) {
    targetDate = targetDate.add(1, 'days');
    console.log(`Date: ${targetDate}`);

    try {
      const result = await getPriceData(coin, targetDate.format("YYYY-MM-DD"));
      dataList = dataList.concat(result);
    } catch(e) {
      console.error(e);
    }

    if (targetDate.isAfter(nowDate)) {
      break;
    }
  }

  writeData(coin, dataList);
}

const code = process.argv[2];

main(code);