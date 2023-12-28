import http from 'k6/http';
import { check } from 'k6';

export const options = {
    scenarios: {
        default: {
            executor: "ramping-arrival-rate",
            timeUnit: "1m",
            startRate: 100,
            preAllocatedVUs: 2,
            stages: [
                {
                    target: 100,
                    duration: "1m"
                },
                {
                    target: 200,
                    duration: "1m"
                }
            ]
        }
    }
};

export default function () {
    const res = http.get(`http://${ __ENV.MY_HOSTNAME }/`);
    check(res, {
        "is status 200": (r) => r.status === 200
    });
}
